"""Paired frozen-checkpoint comparison of rollout-local SPARK corrections."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from helper.get_QAs import get_QAs_from_path
from helper.rebuttal_metrics import CSVLog, asset_id, digest_rows, json_write, runtime_environment
from helper.shared_adapter import preflight_shared_adapter
from helper.specualtive_generate import speculative_generate
from helper.static_correction import STATIC_DEFAULTS, validate_static_correction
from scripts.benchmark_opd_reflex import ROOT, collator, release_runtime_cache
from scripts.benchmark_rebuttal import draft_file, load_frozen_model, project_tensor


STAGES = ('early', 'middle', 'late')
ARMS = ('fastgrpo', 'no_correction', 'token_freq', 'context_cache', 'spark')
STATIC_ARMS = ARMS[2:4]
DIAGNOSTICS = ('opd_static_observations', 'opd_static_nonzero_counts',
               'opd_static_cache_lookups', 'opd_static_cache_hits',
               'opd_static_cache_collisions')


class StopExperiment(RuntimeError):
    """A required input or experimental sanity check failed."""


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run', type=Path, help='SPARK run containing checkpoints/{draft,target}/stepN')
    source.add_argument('--stage', nargs='+', action='append', metavar='ARG',
                        help='repeat for early, middle, late: --stage NAME DRAFT [TARGET_ADAPTER]')
    p.add_argument('--inspect-stages', action='store_true', help='check stage files and projector without running CUDA')
    p.add_argument('--target-adapter', help='default adapter for explicit stages with no adapter argument')
    p.add_argument('--target-model', default='/workspace/storage-shared/models/Qwen2.5-3B-Instruct')
    p.add_argument('--dataset-path', default='/workspace/storage-shared/nlp/users/minhpn19/data/gsm8k/main/train-00000-of-00001.parquet')
    p.add_argument('--output', type=Path)
    for name, default in dict(batch_size=8, responses=8, max_length=2048, max_prompt_length=2048, tuning_prompts=32,
                              evaluation_prompts=64, seed=42, prompt_seed=42, bootstrap_seed=42,
                              bootstrap_samples=2000).items():
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    grids = dict(alphas='0.25,0.5,1,2,4', count_modes='miss,all', orders='1,2',
                 slots='1,4', bonuses='1,2,4,8')
    for name, default in grids.items():
        p.add_argument('--' + name.replace('_', '-'), default=default)
    args = p.parse_args(argv)
    for name, cast in zip(grids, (float, str, int, int, float)):
        try:
            setattr(args, name, sorted({cast(value.strip()) for value in getattr(args, name).split(',')}))
        except ValueError as error:
            p.error(f'{name}: {error}')
    try:
        for arm in STATIC_ARMS:
            for settings in grid(args, arm):
                validate_static_correction('opd_reflex', 0, STATIC_DEFAULTS | settings)
    except ValueError as error:
        p.error(str(error))
    if min(args.batch_size, args.responses, args.tuning_prompts, args.evaluation_prompts,
           args.bootstrap_samples, args.max_length, args.max_prompt_length) < 1:
        p.error('prompt counts, batch size, responses, max length and bootstrap samples must be positive')
    if args.tuning_prompts % args.batch_size or args.evaluation_prompts % args.batch_size:
        p.error('the tuning and evaluation prompt counts must be multiples of the batch size')
    if args.batch_size * args.responses > 256:
        p.error('verification capacity 512 requires at least two nodes per response')
    if args.source_run and args.target_adapter:
        p.error('--source-run uses the target adapter saved at each draft step')
    return args


def adapter_exists(path):
    return (path / 'adapter_config.json').is_file() and any(
        (path / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin'))


def resolve_stages(args):
    if args.source_run:
        drafts = []
        for root in (args.source_run / 'checkpoints', args.source_run):
            for draft in (root / 'draft').glob('step*.pth'):
                match = re.fullmatch(r'step(\d+)', draft.stem)
                if match:
                    drafts.append((int(match[1]), draft, root / 'target' / draft.stem))
            if drafts:
                break
        pairs = sorted((step, draft, adapter) for step, draft, adapter in drafts
                       if step > 0 and adapter_exists(adapter))
        if len(pairs) < 3:
            raise StopExperiment(f'{args.source_run}: need three positive-step target/draft pairs. '
                                 f'Found paired steps {[row[0] for row in pairs]}; '
                                 f'draft steps on disk {sorted(row[0] for row in drafts)}.')
        choice = [pairs[0], pairs[len(pairs) // 2], pairs[-1]]
        stages = {name: (draft, adapter) for name, (_, draft, adapter) in zip(STAGES, choice)}
    else:
        stages = {}
        for entry in args.stage:
            if len(entry) not in (2, 3) or entry[0] in stages:
                raise StopExperiment('use one --stage NAME DRAFT [TARGET_ADAPTER] for each stage')
            name, draft, *adapter = entry
            adapter = adapter[0] if adapter else args.target_adapter
            stages[name] = (draft_file(draft), Path(adapter) if adapter else None)
        if set(stages) != set(STAGES):
            raise StopExperiment('supply exactly the early, middle and late stages')
    if len({draft.resolve() for draft, _ in stages.values()}) != len(STAGES):
        raise StopExperiment('need three distinct draft checkpoints')
    for name, (draft, adapter) in stages.items():
        if not draft.is_file() or (adapter is not None and not adapter_exists(adapter)):
            raise StopExperiment(f'{name}: missing draft or adapter: {draft}, {adapter}')
        match = re.fullmatch(r'step(\d+)', draft.stem)
        target_step = re.fullmatch(r'step(\d+)', adapter.name) if adapter else None
        if match and target_step and int(match[1]) != int(target_step[1]):
            raise StopExperiment(f'{name}: target and draft steps differ: {adapter}, {draft}')
        try:
            state = torch.load(draft, map_location='meta', weights_only=True, mmap=True)['draft_model']
            projector = project_tensor('', state)
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
            raise StopExperiment(f'{name}: cannot verify the saved SPARK projector in {draft}: {error}') from error
        if projector.ndim != 2 or projector.shape[1] != 8:
            raise StopExperiment(f'{name}: expected the saved rank-8 SPARK projector in {draft}')
    return stages


def grid(args, arm):
    if arm == 'token_freq':
        for alpha, mode in itertools.product(args.alphas, args.count_modes):
            yield dict(opd_static_correction=arm, opd_static_alpha=alpha, opd_static_count_mode=mode)
    else:
        for order, slots, bonus in itertools.product(args.orders, args.slots, args.bonuses):
            yield dict(opd_static_correction=arm, opd_cache_order=order, opd_cache_slots=slots, opd_cache_bonus=bonus)


def extend_grid(args, arm, best):
    added = {}
    fields = [('opd_static_alpha', 'alphas')] if arm == 'token_freq' else [('opd_cache_bonus', 'bonuses')]
    for key, attr in fields:
        values = getattr(args, attr)
        if best[key] == values[0]:
            value = values[0] / 2
        elif best[key] == values[-1]:
            value = values[-1] * 2
        else:
            continue
        if value > 0 and math.isfinite(value):
            values[:] = sorted(set(values + [value]))
            added[attr] = value
    # Context order is defined only for n=1 and n=2. Cache slots have a floor of one.
    if arm == 'context_cache':
        if best['opd_cache_slots'] == args.slots[-1]:
            value = args.slots[-1] * 2
        elif best['opd_cache_slots'] == args.slots[0] > 1:
            value = max(1, args.slots[0] // 2)
        else:
            return added
        args.slots[:] = sorted(set(args.slots + [value]))
        added['slots'] = value
    return added


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.


def sums(rows):
    fields = ('accepted', 'rounds', 'accepted_draft', 'proposed_draft', 'generated', 'seconds', *DIAGNOSTICS)
    counters = {key: sum(row[key] for row in rows) for key in fields}
    return dict(**counters, aal=ratio(counters['accepted'], counters['rounds']),
                draft_acceptance_rate=ratio(counters['accepted_draft'], counters['proposed_draft']),
                throughput_tokens_s=ratio(counters['generated'], counters['seconds']),
                cache_hit_rate=ratio(counters['opd_static_cache_hits'], counters['opd_static_cache_lookups']),
                cache_collision_rate=ratio(counters['opd_static_cache_collisions'], counters['opd_static_observations']))


def paired_aal(prompt_totals, ids, draws):
    values = np.asarray([prompt_totals[index] for index in ids], dtype=np.int64)[draws].sum(axis=1)
    return values[:, 0] / np.maximum(values[:, 1], 1)


def replay(args, stage, arm, settings, phase, model, tokenizer, batches, logs, warmup=False):
    method = 'fastgrpo' if arm == 'fastgrpo' else 'opd_reflex'
    common = dict(stage=stage, arm=arm, phase=phase, settings=settings)
    release_runtime_cache(model)
    torch.cuda.empty_cache()

    def run(batch, seed):
        torch.manual_seed(seed)
        return speculative_generate(model, batch['input_ids'].cuda(), batch['attention_mask'].cuda(),
            tokenizer, method=method, do_sample=True, repeated_generate_nums=args.responses,
            temperature=1., top_p=.95, max_length=args.max_length, max_draft_k=8,
            min_draft_token_length=3, max_draft_token_length=5, draft_token_length_c=.75,
            verification_capacity=512, max_verification_num=160, generation_profile='clean',
            opd_rank=8, opd_topk=16, opd_fast_lr=.01, opd_visited_weight=1., opd_frontier_weight=1.,
            opd_b_sharing_scope='per_prompt_group', opd_b_reset_mode='rollout', opd_backend='triton',
            opd_train_projector=False, opd_update_interval_rounds=int(arm == 'spark'), **settings)

    if warmup:
        for _, seed, batch in batches:
            run(batch, seed)
    rollouts = []
    prompts = defaultdict(lambda: [0, 0])
    for indices, seed, batch in batches:
        out = run(batch, seed)
        row = dict(**common, seed=seed, prompt_ids=indices, accepted=out['total_acc_length'],
                   rounds=out['total_decoded_token_num'], accepted_draft=out['total_accepted_draft_tokens'],
                   proposed_draft=out['total_proposed_draft_tokens'],
                   generated=sum(out['response_generated_tokens']), seconds=out['total_time_cost'],
                   proposal_profile=out.get('opd_proposal_profile_effective'),
                   **{key: out.get(key, 0) for key in DIAGNOSTICS})
        rollouts.append(row)
        logs['rollouts'].write(row)
        for index, (accepted, rounds, length) in enumerate(zip(
                out['response_accepted_length_sum'], out['response_verification_rounds'],
                out['response_generated_tokens'], strict=True)):
            prompt = indices[index // args.responses]
            prompts[prompt][0] += accepted
            prompts[prompt][1] += rounds
            logs['responses'].write(json.dumps(dict(**common, seed=seed, prompt_index=prompt,
                response=index % args.responses, accepted_length_sum=accepted,
                verification_rounds=rounds, generated_tokens=length)) + '\n')
        logs['responses'].flush()
    measured = dict(**common, **sums(rollouts), proposal_profile=rollouts[0]['proposal_profile'])
    print(f'{phase} {stage} {arm} {settings}: AAL={measured["aal"]:.6f}', flush=True)
    return measured, prompts


def search_grid(args, arm, phase, run, extension):
    tested = {}
    for _ in range(2 if extension is not None else 1):
        for settings in grid(args, arm):
            key = tuple(settings.items())
            if key not in tested:
                tested[key], _ = run(arm, settings, phase)
        best = max(tested.values(), key=lambda row: row['aal'])
        if extension is None or arm in extension:
            break
        extension[arm] = extend_grid(args, arm, best['settings'])
    return list(tested.values()), best


def summarize_stage(results, ids, draws, logs, summary):
    means = {arm: paired_aal(prompts, ids, draws) for arm, (_, prompts) in results.items()}
    reference = results['no_correction'][0]['aal']
    gain = results['spark'][0]['aal'] - reference
    for arm, (measured, _) in results.items():
        row = dict(measured, share_of_spark_gain=(measured['aal'] - reference) / gain if gain else None)
        for baseline in ('no_correction', 'spark'):
            ci = np.percentile(means[arm] - means[baseline], [2.5, 97.5])
            row.update({f'delta_{baseline}': measured['aal'] - results[baseline][0]['aal'],
                        f'ci95_{baseline}_low': float(ci[0]), f'ci95_{baseline}_high': float(ci[1])})
        summary.append(row)
        logs['summary'].write(row)
    if gain <= 0:
        raise StopExperiment(f'{results["spark"][0]["stage"]}: SPARK AAL did not exceed the no-correction engine.')


def write_report(output, config, summary, choices, oracle, extension, error):
    def interval(row, reference):
        return f'{row[f"delta_{reference}"]:+.4f} [{row[f"ci95_{reference}_low"]:+.4f}, {row[f"ci95_{reference}_high"]:+.4f}]'

    report = ['# Online correction on frozen SPARK checkpoints', '',
              f'Status: {error or "complete"}', '',
              'Corrections were added to bf16 draft logits. Miss mode counts the last emitted token '
              'when its node has a draft-head context and no matching child in the verified tree. '
              'Cache collisions count inserts that replace a different full key in an occupied bucket. '
              'Collision rate divides those replacements by all cache inserts. Order 2 uses 2²⁰ buckets per group.',
              '', 'Throughput uses clean generation wall time. The static corrections use tensor operations '
              'and SPARK uses specialized kernels. Each evaluation arm and stage has one full warm-up pass. '
              'Grid trials use measured AAL for selection. Their times include any first-use overhead.',
              '', '| Stage | Arm | AAL | Δ vs no correction (95% CI) | Δ vs SPARK (95% CI) | Share of SPARK gain | '
              'Draft acceptance | Tokens/s | Cache lookups | Hit rate | Collision rate | Nonzero group/token counts |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    order = sorted(summary, key=lambda row: (STAGES.index(row['stage']), ARMS.index(row['arm'])))
    for row in order:
        share = 'N/A' if row['share_of_spark_gain'] is None else f'{row["share_of_spark_gain"]:.2%}'
        report.append(f'| {row["stage"]} | {row["arm"]} | {row["aal"]:.4f} | '
            f'{interval(row, "no_correction")} | {interval(row, "spark")} | {share} | '
            f'{row["draft_acceptance_rate"]:.2%} | {row["throughput_tokens_s"]:.2f} | '
            f'{row["opd_static_cache_lookups"]} | {row["cache_hit_rate"]:.2%} | '
            f'{row["cache_collision_rate"]:.2%} | {row["opd_static_nonzero_counts"]} |')
    profiles = sorted({str(row['proposal_profile'] or 'uncalibrated fallback') for row in summary if row['arm'] == 'spark'})
    report.extend(['', f'SPARK proposal profiles: {", ".join(profiles)}', '',
                   '## Selected on the middle-stage tuning prompts', '', '~~~json',
                   json.dumps(choices, indent=2), '~~~', '', 'Grid extensions', '', '~~~json',
                   json.dumps(extension, indent=2), '~~~', '', '## Late-stage oracle-tuned upper bounds', '',
                   'The configurations below were selected on the evaluation prompts.', '', '~~~json',
                   json.dumps(oracle, indent=2), '~~~', '', '## Reproduction', '',
                   f'`{config["command"]}`', '',
                   f'Commit: {config.get("environment", {}).get("git_commit")}', '',
                   '| Stage | Draft | Draft SHA-256 | Target adapter | Adapter SHA-256 | Projector SHA-256 |',
                   '|---|---|---|---|---|---|'])
    for stage, files in config.get('inputs', {}).get('stages', {}).items():
        adapter = files['adapter'] or {}
        report.append(f'| {stage} | `{files["draft"]["resolved_path"]}` | {files["draft"]["sha256"]} | '
                      f'`{adapter.get("resolved_path", "base target")}` | {adapter.get("sha256", "")} | '
                      f'{files.get("projector_sha256", "")} |')
    report.extend(['', 'See config_resolved.yaml for the remaining input hashes and prompt IDs. '
                   'See tuning_grid.csv, summary.csv and logs/ for all measurements.'])
    (output / 'report.md').write_text('\n'.join(report) + '\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.inspect_stages:
        try:
            print(json.dumps(resolve_stages(args), default=str, indent=2))
        except StopExperiment as error:
            raise SystemExit(f'STOP: {error}') from None
        return

    output = args.output or ROOT / 'outputs/rebuttal/online_correction' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    with (output / 'logs/response_metrics.jsonl').open('x', buffering=65536) as response_log:
        logs = dict(responses=response_log, summary=CSVLog(output / 'summary.csv'),
                    grid=CSVLog(output / 'tuning_grid.csv'), rollouts=CSVLog(output / 'logs/rollout_metrics.csv'))
        try:
            execute(args, argv, output, logs)
        finally:
            for key in ('summary', 'grid', 'rollouts'):
                logs[key].close()


def execute(args, argv, output, logs):
    from transformers import AutoTokenizer

    os.environ.setdefault('OPD_SAMPLER_MODE', 'finite')
    config = dict(args=json.loads(json.dumps(vars(args), default=str)),
                  command=shlex.join([sys.executable, 'scripts/benchmark_online_correction.py', *argv]),
                  settings=dict(batch_size=args.batch_size, responses=args.responses, dtype='bf16',
                                temperature=1., top_p=.95, max_length=args.max_length,
                                verification_capacity=512, max_verification_num=160,
                                max_draft_k=8, draft_depth=[3, 5], draft_token_length_c=.75,
                                spark_rank=8, spark_topk=16, spark_fast_lr=.01, spark_update_stream=True,
                                spark_interval=1, spark_weights=[1., 1.], sharing_scope='per_prompt_group'),
                  proposal_env={key: value for key, value in os.environ.items() if key.startswith('OPD_')})
    summary, choices, oracle, extension = [], {}, {}, {}
    error = None
    try:
        stages = resolve_stages(args)
        if torch.cuda.device_count() != 1 or int(os.getenv('WORLD_SIZE', '1')) != 1:
            raise StopExperiment('select exactly one visible CUDA GPU')
        device = torch.device('cuda:0')
        config['environment'] = runtime_environment(ROOT, device)
        json_write(output / 'environment.json', config['environment'])
        config['inputs'] = dict(target=asset_id(args.target_model), dataset=asset_id(args.dataset_path),
            stages={name: dict(draft=asset_id(draft), adapter=asset_id(adapter) if adapter else None)
                    for name, (draft, adapter) in stages.items()})
        rows = get_QAs_from_path(args.dataset_path, 'train')
        needed = args.tuning_prompts + args.evaluation_prompts
        if len(rows) < needed:
            raise StopExperiment(f'need {needed} distinct prompts, found {len(rows)}')
        order = list(range(len(rows)))
        random.Random(args.prompt_seed).shuffle(order)
        config.update(tuning_prompt_ids=order[:args.tuning_prompts], eval_prompt_ids=order[args.tuning_prompts:needed],
                      prompt_rows_sha256=digest_rows([rows[index] for index in order[:needed]]))
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, padding_side='left', local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        make_batch = collator(tokenizer, args.max_prompt_length)
        batches = []
        for offset in range(0, needed, args.batch_size):
            indices = order[offset:offset + args.batch_size]
            batch = make_batch([rows[index] for index in indices])
            if batch['input_ids'].shape[-1] >= args.max_length:
                raise StopExperiment('a prompt exhausts the total context cap')
            batches.append((indices, args.seed + offset // args.batch_size, batch))
        split = args.tuning_prompts // args.batch_size
        draws = np.random.default_rng(args.bootstrap_seed).integers(
            args.evaluation_prompts, size=(args.bootstrap_samples, args.evaluation_prompts))
        with torch.inference_mode():
            for stage in ('middle', 'early', 'late'):
                draft, adapter = stages[stage]
                state = torch.load(draft, map_location='cpu', weights_only=True, mmap=True)['draft_model']
                projector = state.pop('opd_projector')
                if not torch.isfinite(projector).all():
                    raise StopExperiment(f'{stage}: nonfinite learned projector')
                config['inputs']['stages'][stage]['projector_sha256'] = hashlib.sha256(
                    projector.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                preflight_shared_adapter(adapter, args.target_model)
                model = load_frozen_model(args.target_model, adapter, state, device, torch.bfloat16, 'sdpa')
                del state
                model.enable_opd(8)
                model.opd_projector.requires_grad_(False)

                def run(arm, settings, phase):
                    return replay(args, stage, arm, settings, phase, model, tokenizer,
                                  batches[:split] if phase == 'tuning' else batches[split:],
                                  logs, warmup=phase == 'evaluation')

                if stage == 'middle':
                    baseline, _ = run('no_correction', {}, 'tuning')
                    logs['grid'].write(dict(baseline, selected=False))
                    for arm in STATIC_ARMS:
                        trials, best = search_grid(args, arm, 'tuning', run, extension)
                        for row in trials:
                            logs['grid'].write(dict(row, selected=row is best))
                        choices[arm] = best
                        if best['aal'] < baseline['aal']:
                            raise StopExperiment(f'{arm} was below no correction at every tuning grid point.')
                results = {}
                for arm in ARMS:
                    if arm == 'spark':
                        model.load_opd_projector(projector)
                    results[arm] = run(arm, choices.get(arm, {}).get('settings', {}), 'evaluation')
                summarize_stage(results, config['eval_prompt_ids'], draws, logs, summary)
                if stage == 'late':
                    for arm in STATIC_ARMS:
                        trials, best = search_grid(args, arm, 'late_oracle', run, None)
                        for row in trials:
                            logs['grid'].write(dict(row, selected=row is best))
                        oracle[arm] = best
                        if best['aal'] < results['no_correction'][0]['aal']:
                            raise StopExperiment(f'{arm} was below no correction at every late oracle grid point.')
                release_runtime_cache(model)
                del model, projector
                torch.cuda.empty_cache()
    except StopExperiment as exc:
        error = f'STOP: {exc}'
        raise SystemExit(error) from None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        config.update(effective_grids={arm: list(grid(args, arm)) for arm in STATIC_ARMS},
                      grid_extensions=extension, status=error or 'complete')
        json_write(output / 'config_resolved.yaml', config)
        write_report(output, config, summary, choices, oracle, extension, error)
        print(f'Report: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
