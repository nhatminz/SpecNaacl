"""Steady-state frozen FastGRPO / SPARK throughput and diagnostic replays."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch

from helper.get_QAs import get_QAs_from_path
from helper.rebuttal_metrics import CSVLog, asset_id, digest_rows, json_write, length_metrics, runtime_environment, utc_now
from helper.specualtive_generate import speculative_generate
from scripts.benchmark_opd_reflex import ROOT, collator, release_runtime_cache


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target-model', 'model-key', 'draft-checkpoint', 'dataset-path', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sweep', choices=('concurrency', 'prompt_groups', 'long_context'), required=True)
    parser.add_argument('--target-adapter', default='')
    parser.add_argument('--projector-checkpoint', default='')
    parser.add_argument('--generation-profile', choices=('clean', 'detailed'), default='detailed')
    parser.add_argument('--opd-b-sharing-scope', choices=('per_prompt_group', 'rollout_batch'), default='per_prompt_group')
    parser.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--attn-implementation', default='sdpa')
    parser.add_argument('--opd-update-stream', type=int, choices=(0, 1), default=1)
    for name, default in dict(batch_size=8, responses=8, max_length=2048, max_prompt_length=2048,
                              verification_capacity=512, max_verification_num=160, max_draft_k=8,
                              max_draft_token_length=5, min_draft_token_length=3, rank=8, topk=16,
                              warmup=1, repetitions=5, seed=42, prompt_seed=42, round_log_frequency=20,
                              top_k=0).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    for name, default in dict(temperature=1., top_p=.95, draft_token_length_c=.75,
                              opd_fast_lr=.01, opd_visited_weight=1., opd_frontier_weight=1.).items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    args = parser.parse_args(argv)
    if min(args.warmup, args.repetitions, args.batch_size, args.responses, args.max_length,
           args.max_prompt_length, args.verification_capacity, args.max_verification_num) < 1:
        parser.error('warmup, repetitions, lengths, sizes, and capacity must be positive')
    if args.opd_fast_lr <= 0 or args.topk < args.max_draft_k or args.round_log_frequency < 0:
        parser.error('SPARK requires nonzero fast LR, topk >= max draft k, and a nonnegative log frequency')
    return args


def configurations(args):
    if args.sweep == 'concurrency':
        return [(8, responses, None) for responses in (4, 8, 16)]
    if args.sweep == 'prompt_groups':
        return [(prompts, 8, None) for prompts in (1, 2, 4, 8, 16)]
    return [(args.batch_size, args.responses, length) for length in (2048, 4096)]


def draft_file(path):
    path = Path(path)
    return path / 'draft.pth' if path.is_dir() else path


def project_tensor(source, draft_state):
    if source:
        payload = torch.load(draft_file(source), map_location='cpu', weights_only=True)
        value = payload if torch.is_tensor(payload) else payload.get('draft_model', payload).get('opd_projector')
    else:
        value = draft_state.get('opd_projector')
    if not torch.is_tensor(value):
        raise ValueError('trained SPARK A missing. Set PROJECTOR_CHECKPOINT to a learned draft.pth or tensor checkpoint')
    return value


def repetition_summary(rows, fields):
    result = {'repetitions': len(rows)}
    for key in fields:
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key + '_mean'] = statistics.mean(values) if values else None
        result[key + '_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
    return result


STEADY_FIELDS = (
    'generation_wall_time_s', 'throughput_tokens_s', 'aal', 'acceptance_rate', 'generated_tokens',
    'length_mean', 'length_max', 'fraction_over_2048', 'fraction_reaching_4096',
    'peak_allocated_bytes', 'peak_reserved_bytes', 'correction_matrix_count', 'correction_matrix_bytes',
    'dense_fp32_correction_bytes', 'projector_bytes', 'correction_workspace_bytes', 'spark_updates',
)


def main(argv):
    args = parse_args(argv)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helper.fastgrpo_model import FastGRPOModel
    from helper.shared_adapter import preflight_shared_adapter

    if torch.cuda.device_count() != 1 or int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('select exactly one visible CUDA GPU')
    started = time.perf_counter()
    start_date = utc_now()
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    device = torch.device('cuda:0')
    preflight_shared_adapter(args.target_adapter, args.target_model)
    checkpoint = draft_file(args.draft_checkpoint).resolve(strict=True)
    draft_state = torch.load(checkpoint, map_location='cpu', weights_only=True)['draft_model']
    projector = project_tensor(args.projector_checkpoint, draft_state)
    draft_state.pop('opd_projector', None)
    target = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=dtype,
        attn_implementation=args.attn_implementation, local_files_only=True).to(device).eval()
    config = deepcopy(target.config)
    config.rope_scaling = None
    config.num_hidden_layers = 1
    model = FastGRPOModel(config, target).to(device).eval()
    model.draft_model.load_state_dict(draft_state)
    del draft_state
    if args.target_adapter:
        model.target_model = PeftModel.from_pretrained(model.target_model, args.target_adapter).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = get_QAs_from_path(args.dataset_path, 'train')
    shapes = configurations(args)
    largest_concurrency = max(batch*responses for batch,responses,_ in shapes)
    if min(args.verification_capacity // largest_concurrency,args.max_verification_num) < 2:
        raise ValueError('verification budget must allow at least two nodes per response at the largest requested concurrency')
    needed = args.repetitions * max(batch for batch, _, _ in shapes)
    if len(rows) < needed:
        raise ValueError(f'benchmark requires {needed} rows for the paired prompt schedule, found {len(rows)}')
    order = list(range(len(rows)))
    random.Random(args.prompt_seed).shuffle(order)
    group_stride = max(batch for batch, _, _ in shapes)
    make_batch = collator(tokenizer, args.max_prompt_length)

    output = Path(args.output)
    output.mkdir(parents=True)
    logs = output / 'logs'
    logs.mkdir()
    env = runtime_environment(ROOT, device)
    json_write(output / 'environment.json', env)
    inputs = dict(target=asset_id(args.target_model), draft=asset_id(checkpoint),
                  projector=asset_id(args.projector_checkpoint or checkpoint), dataset=asset_id(args.dataset_path),
                  target_adapter=asset_id(args.target_adapter) if args.target_adapter else None)
    config = dict(args=vars(args), inputs=inputs, dataset_sample_count=len(rows),
                  prompt_ids=order[:needed], prompt_rows_sha256=digest_rows([rows[i] for i in order[:needed]]),
                  projector_sha256=hashlib.sha256(projector.contiguous().numpy().tobytes()).hexdigest(),
                  proposal_env={key: value for key, value in os.environ.items() if key.startswith('OPD_')})
    json_write(output / 'config_resolved.yaml', config)
    rollout_log = CSVLog(logs / 'rollout_metrics.csv')
    timing_log = CSVLog(logs / 'generation_profile.csv') if args.generation_profile == 'detailed' else None
    memory_log = CSVLog(logs / 'memory_profile.csv') if timing_log else None
    round_log = CSVLog(logs / 'verification_rounds.csv') if timing_log and args.round_log_frequency else None
    raw = []
    diagnostic_rows = []
    by_method_and_shape = {}
    warmup_time = diagnostic_time = 0.
    with torch.inference_mode(), (logs / 'response_metrics.jsonl').open('x', buffering=65536) as responses:
        for method in ('fastgrpo', 'opd_reflex'):
            if method == 'opd_reflex':
                model.enable_opd(args.rank, projector=projector)
                model.opd_projector.requires_grad_(False)
            for batch_size, responses_per_prompt, cap in shapes:
                release_runtime_cache(model)
                torch.cuda.empty_cache()
                prompt_ids = [order[i*group_stride:i*group_stride+batch_size] for i in range(args.repetitions)]
                batches = [make_batch([rows[index] for index in ids]) for ids in prompt_ids]
                if cap is None and any(batch['input_ids'].shape[-1] >= args.max_length for batch in batches):
                    raise ValueError('a prompt exhausts the legacy total-context cap')

                def run(rep, mode):
                    batch = batches[rep]
                    seed = args.seed + rep
                    torch.manual_seed(seed)
                    return speculative_generate(model, batch['input_ids'].to(device), batch['attention_mask'].to(device),
                        tokenizer, do_sample=True, method=method, repeated_generate_nums=responses_per_prompt,
                        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k or None,
                        max_length=args.max_length, max_response_tokens=cap, max_draft_k=args.max_draft_k,
                        max_draft_token_length=args.max_draft_token_length, min_draft_token_length=args.min_draft_token_length,
                        verification_capacity=args.verification_capacity, max_verification_num=args.max_verification_num,
                        draft_token_length_c=args.draft_token_length_c, generation_profile=mode,
                        round_log_frequency=args.round_log_frequency, opd_rank=args.rank, opd_topk=args.topk,
                        opd_fast_lr=args.opd_fast_lr, opd_visited_weight=args.opd_visited_weight,
                        opd_frontier_weight=args.opd_frontier_weight, opd_update_stream=bool(args.opd_update_stream),
                        opd_train_projector=False, opd_update_interval_rounds=1, opd_b_reset_mode='rollout',
                        opd_b_sharing_scope=args.opd_b_sharing_scope, opd_backend='triton')

                warm_started = time.perf_counter()
                for _ in range(args.warmup):
                    for rep in range(args.repetitions):
                        run(rep, 'clean')
                warmup_time += time.perf_counter() - warm_started
                settings = dict(method=method, model=args.model_key, dataset=args.dataset, sweep=args.sweep,
                                batch_size=batch_size, responses_per_prompt=responses_per_prompt,
                                initial_concurrency=batch_size*responses_per_prompt, response_cap=cap,
                                legacy_total_cap=args.max_length if cap is None else None)
                measured = []
                clean_paths = []
                for rep in range(args.repetitions):
                    out = run(rep, 'clean')
                    generated = out['generated_token_ids']
                    lengths = out['response_generated_tokens']
                    clean_paths.append(digest_rows(generated))
                    common = dict(settings, repetition=rep+1, seed=args.seed+rep)
                    wall = out['total_time_cost']
                    row = dict(**common, **length_metrics(lengths, generated, tokenizer.eos_token_id,
                        args.max_length, cap, out['prompt_token_lengths'], responses_per_prompt),
                        **out['generation_memory'], generation_wall_time_s=wall,
                        throughput_tokens_s=sum(lengths)/wall,
                        aal=out['total_acc_length']/max(1, out['total_decoded_token_num']),
                        acceptance_rate=out['draft_acceptance_rate'], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), spark_updates=out.get('opd_updates',0),
                        prompt_indices=prompt_ids[rep])
                    rollout_log.write(row)
                    measured.append(row)
                    for index,length in enumerate(lengths):
                        responses.write(json.dumps(dict(**common, prompt_index=prompt_ids[rep][index//responses_per_prompt],
                            response=index%responses_per_prompt, generated_tokens=length,
                            accepted_length_sum=out['response_accepted_length_sum'][index],
                            verification_rounds=out['response_verification_rounds'][index]))+'\n')
                raw.append(dict(**settings, **repetition_summary(measured, STEADY_FIELDS)))
                by_method_and_shape[method,batch_size,responses_per_prompt,cap] = measured
                if timing_log:
                    # Measure the diagnostic path separately after clean measurements.
                    diagnostic_started = time.perf_counter()
                    release_runtime_cache(model)
                    torch.cuda.empty_cache()
                    for rep in range(args.repetitions):
                        run(rep, 'detailed')
                    diagnostics = []
                    for rep in range(args.repetitions):
                        profile = run(rep, 'detailed')
                        if digest_rows(profile['generated_token_ids']) != clean_paths[rep]:
                            raise RuntimeError(f'diagnostic replay changed generated tokens for {settings}, rep={rep+1}')
                        common = dict(settings, repetition=rep+1, seed=args.seed+rep)
                        timing_log.write(dict(**common, opd_backend=profile.get('opd_backend','off'),
                            proposal_profile=profile.get('opd_proposal_profile_effective'), **profile['generation_profile']))
                        memory_log.write(dict(**common, **profile['generation_memory'],
                                              spark_updates=profile.get('opd_updates', 0),
                                              spark_feedback_states=profile.get('opd_selected_states', 0)))
                        diagnostics.append(dict(**profile['generation_profile'], **profile['generation_memory']))
                        if round_log:
                            for row in profile['verification_rounds']:
                                round_log.write(dict(**common, **row))
                    diagnostic_rows.append(dict(**settings, **repetition_summary(diagnostics, diagnostics[0])))
                    diagnostic_time += time.perf_counter() - diagnostic_started
    for log in (rollout_log, timing_log, memory_log, round_log):
        if log:
            log.close()
    summary = CSVLog(output / 'summary.csv')
    for row in raw:
        summary.write(row)
    summary.close()
    paired = []
    for batch, responses, cap in shapes:
        baseline = by_method_and_shape['fastgrpo',batch,responses,cap]
        spark = by_method_and_shape['opd_reflex',batch,responses,cap]
        pairs = [dict(throughput_speedup=b['throughput_tokens_s']/a['throughput_tokens_s'],
                      wall_time_speedup=a['generation_wall_time_s']/b['generation_wall_time_s'])
                 for a,b in zip(baseline,spark)]
        paired.append(dict(batch_size=batch,responses_per_prompt=responses,response_cap=cap,
                           **repetition_summary(pairs,pairs[0])))
    for filename, contents in (('paired_speedups.csv', paired), ('diagnostic_summary.csv', diagnostic_rows)):
        if contents:
            log = CSVLog(output / filename)
            for record in contents:
                log.write(record)
            log.close()
    json_write(output / 'summary.json', dict(started_at=start_date, finished_at=utc_now(),
        absolute_wall_time_s=time.perf_counter()-started, warmup_s=warmup_time,
        diagnostic_replay_s=diagnostic_time, results=raw, paired_speedups=paired, diagnostics=diagnostic_rows))
