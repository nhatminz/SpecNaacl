#!/usr/bin/env python3
"""Aggregate compatible paired seeds at measured updates and observed GPU budgets."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics

BASE_ARGS = '''
batch_size accumulation_steps draft_accumulation_steps grpo_iteration_num num_epochs sample_num
train_data_fraction max_train_samples train_subset_seed prompt_order_seed train_split train_option
max_target_optimizer_steps max_rollout_prompts max_grpo_steps is_train_draft target_lr draft_lr
draft_lr_multiplier beta epsilon repeated_generate_nums temperature top_p max_length max_response_tokens
max_prompt_length max_training_padding_gap max_training_token logps_chunk_size verification_capacity
max_draft_token_length max_draft_k max_verification_num min_draft_token_length draft_token_length_c
dtype attn_implementation generation_profile statistical_time num_workers persistent_workers
'''.split()
OPD_ARGS = '''
opd_rank opd_topk opd_fast_lr opd_visited_weight opd_frontier_weight opd_backend opd_update_stream
opd_train_projector opd_projector_init opd_projector_seed opd_projector_lr opd_update_interval_rounds
opd_b_sharing_scope opd_b_reset_mode kv_gather_strategy
'''.split()
OPD_ENV = '''
OPD_PROPOSAL_MODE OPD_DENSE_IMPLEMENTATION OPD_SAMPLER_MODE OPD_KV_MAX_RETAINED_TOKENS
'''.split()
SPEED_METRICS = ('generation_time_s', 'training_wall_time_s')
METRICS = ('reward', 'correctness', 'accuracy', 'response_length', 'generated_tokens', 'aal',
           'acceptance_rate', 'generation_tokens_s', *SPEED_METRICS)
T = 'cumulative_training_wall_time_s'
U = 'target_optimizer_steps'


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def stat(values):
    values = [value for value in values if value is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def number(row, key):
    value = row.get(key)
    if value in ('', None):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'nonfinite metric {key}')
    return value


def require_fields(payload, keys):
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f'missing compatibility fields {missing}')
    return {key: payload[key] for key in keys}


def signature(config, method, *, ar=False):
    args = config['args']
    base_keys = list(BASE_ARGS)
    if ar:
        # A fixed AR reference has no speculative budget or online drafter.
        base_keys = [key for key in base_keys if not key.startswith(('draft_', 'verification_', 'max_verification',
                     'max_draft_', 'min_draft_', 'generation_profile', 'is_train_draft'))]
    shared = require_fields(args, base_keys)
    if not ar:
        shared['draft_initialization_mode'] = args.get('draft_initialization_mode','pretrained')
    shared['assets'] = {}
    for name in ('target', 'target_adapter', 'train_dataset', 'draft'):
        if name == 'draft' and ar:
            continue
        source = config['inputs'][name]
        optional = name == 'target_adapter' or (
            name == 'draft' and shared.get('draft_initialization_mode') == 'random')
        if not source and not optional:
            raise ValueError(f'missing stable {name} identity')
        shared['assets'][name] = source['sha256'] if source else None
    shared['dataset'] = require_fields(config['dataset'],
        ('selected_rows_sha256','selected_sample_count','prompt_order_protocol'))
    shared['hardware'] = require_fields(config['environment'], ('gpu','gpu_memory_bytes','world_size','gpu_count'))
    if shared['hardware']['gpu_count'] != 1 or shared['hardware']['world_size'] != 1:
        raise ValueError('aggregation expects data measured on one GPU')
    shared['runtime'] = require_fields(config['environment'], ('torch_version','cuda_build','driver'))
    shared['software'] = require_fields(config['environment']['software'],
                                       ('transformers','peft','triton','math-verify'))
    shared['basis'] = config.get('metric_protocol', 'rebuttal_v1')
    method_specific = {}
    if method == 'opd_reflex':
        method_specific = require_fields(args, OPD_ARGS)
        proposal = config['proposal_env']
        method_specific.update({key: proposal.get(key) for key in OPD_ENV})
        method_specific['proposal_profile_sha256'] = config.get('proposal_profile_sha256')
        method_specific['opd_initial_a_sha256'] = config['initialization'].get('opd_initial_a_sha256')
        if not method_specific['opd_initial_a_sha256']:
            raise ValueError('missing SPARK projector initialization identity')
    return shared, method_specific


def legacy_updates(run_dir, metadata, config, summary):
    """Join the original rollout CSV to the actual optimizer-call cadence."""
    rows = read_csv(run_dir / metadata['legacy_rollout_timing'])
    times = {int(row[U]): row[T] for row in read_csv(run_dir / metadata['comparable_wall_clock_csv'])} if (
        metadata.get('comparable_wall_clock_csv')) else {}
    cadence = {}
    for prompts, step in summary.get('optimizer_step_cadence', []):
        cadence[int(prompts)] = int(step)
    if not cadence or not rows:
        raise ValueError('legacy import requires optimizer_step_cadence and rollout_timing.csv')
    prompts = last_update = 0
    result = []
    for row in rows:
        wall = number(row,'iter_generation_time_s')
        if wall is None or wall <= 0:
            continue
        prompts += int(row['total_prompts'])
        update = cadence.get(prompts)
        if update is None or update == last_update:
            continue
        tokens = int(row['cumulative_rollout_tokens'])
        gen = float(row['cumulative_generation_time_s']) if metadata.get('comparable_generation_time') else None
        result.append({U: update, T: times.get(update),
            'cumulative_generated_tokens':tokens,
            'cumulative_responses':prompts*int(config['args']['repeated_generate_nums']),
            'cumulative_aal':row['cumulative_aal'], 'cumulative_acceptance_rate':row['cumulative_acceptance_rate'],
            'cumulative_generation_time_s':gen, 'cumulative_generation_tokens_s':tokens/gen if gen else None})
        # The cadence records the last actual update for a rollout. Earlier
        # updates at this boundary cannot be reconstructed from old logs.
        last_update = update
    return result


@dataclass
class Run:
    path: str
    method: str
    seed: int
    config: dict
    final_updates: int
    final_wall: object
    updates: dict
    evaluations: list
    eval_signature: object
    shared: dict
    specific: dict

    @classmethod
    def load(cls, run_dir, historical=None):
        logs = run_dir / 'logs'
        if historical is None:
            config = read_json(run_dir / 'config_resolved.yaml')
            summary = read_json(run_dir / 'summary.json')
            rows = read_csv(logs / 'optimizer_updates.csv')
            evaluation_log = logs
        else:
            config = historical['config']
            if not historical.get('source') or config.get('metric_protocol') != 'rebuttal_v1':
                raise ValueError('historical import requires source provenance and confirmed equivalent metric_protocol')
            summary = read_json(run_dir / historical.get('summary', 'summary.json'))
            rows = (read_csv(run_dir / historical['optimizer_updates']) if 'optimizer_updates' in historical
                    else legacy_updates(run_dir,historical,config,summary))
            evaluation_log = run_dir / historical.get('evaluation_dir', 'logs')
        method = config['identifiers']['method']
        seed = int(config['identifiers']['seed'])
        shared,specific = signature(config,method,ar=method=='ar')
        if config['args']['seed'] != seed:
            raise ValueError('seed disagreement in resolved config')
        final_updates = int(summary[U])
        final_wall = number(summary,'training_wall_time_s')
        if final_updates < 1 or (historical is None and not final_wall) or (final_wall is not None and final_wall <= 0):
            raise ValueError('need actual completed updates and positive observed E2E time when available')
        updates = {int(row[U]):row for row in rows}
        if len(updates) != len(rows) or not updates or max(updates) != final_updates:
            raise ValueError('optimizer time series needs unique actual update indices and a row at final update')
        times = [number(updates[index], T) for index in sorted(updates)]
        observed = [value for value in times if value is not None]
        if (len(observed) != len(times) and historical is None or
            any(value < 0 or (final_wall is not None and value > final_wall + 1e-6) for value in observed) or
            observed != sorted(observed)):
            raise ValueError('optimizer wall-clock times must be nondecreasing and within observed training time')
        evaluations = read_csv(evaluation_log / 'evaluation_metrics.csv')
        eval_signature = None
        if evaluations:
            settings = read_json(evaluation_log / 'evaluation_summary.json')['settings']
            eval_signature = dict(settings, target=settings['target']['sha256'], dataset=settings['dataset']['sha256'])
            if not settings['do_sample']:
                eval_signature.pop('batch_size', None)
            for row in evaluations:
                update, time_s = int(row[U]), number(row, T)
                if (not 0 <= update <= final_updates or time_s is None or time_s < 0 or
                    (final_wall is not None and time_s > final_wall + 1e-6)):
                    raise ValueError('evaluation index is outside observed training')
            evaluations.sort(key=lambda row: (float(row[T]), int(row[U])))
        return cls(str(run_dir),method,seed,config,final_updates,final_wall,updates,evaluations,eval_signature,shared,specific)

    def accuracy_at(self, value, axis):
        if axis == 'updates':
            rows = [row for row in self.evaluations if int(row[U]) == value]
        else:
            rows = [row for row in self.evaluations if float(row[T]) <= value]
        return number(rows[-1],'accuracy') if rows else None

    def metrics_at(self, index):
        row = self.updates[index]
        tokens = number(row,'cumulative_generated_tokens')
        responses = number(row,'cumulative_responses')
        return dict(reward=number(row,'cumulative_reward_mean'),
            correctness=number(row,'cumulative_correctness_mean'), accuracy=self.accuracy_at(index,'updates'),
            generated_tokens=tokens,response_length=tokens/responses if responses and tokens is not None else None,
            aal=number(row,'cumulative_aal'),acceptance_rate=number(row,'cumulative_acceptance_rate'),
            generation_tokens_s=number(row,'cumulative_generation_tokens_s'),
            generation_time_s=number(row,'cumulative_generation_time_s'),training_wall_time_s=number(row,T))


def comparison(runs, ar_run=None):
    by_method = defaultdict(dict)
    for run in runs:
        if run.seed in by_method[run.method]:
            raise ValueError(f'duplicate {run.method} run for seed {run.seed}. Select the intended directories explicitly')
        by_method[run.method][run.seed] = run
    seeds = sorted(set(by_method['fastgrpo']).intersection(by_method['opd_reflex']))
    if not seeds:
        raise ValueError('no paired seed has both methods')
    paired = [by_method[method][seed] for method in ('fastgrpo','opd_reflex') for seed in seeds]
    eval_configs = {json.dumps(run.eval_signature, sort_keys=True) for run in paired if run.eval_signature is not None}
    if len(eval_configs) > 1:
        raise ValueError('incompatible held-out evaluation dataset, protocol, or decoding settings')
    exact = sorted(set.intersection(*(set(run.updates) for run in paired)))
    if not exact:
        raise ValueError('no common observed actual target optimizer update')
    matched = exact[-1]
    report, per_seed = [], []
    for method in ('fastgrpo','opd_reflex'):
        individual = []
        for seed in seeds:
            run = by_method[method][seed]
            row = dict(seed=seed, method=method, run_dir=run.path,
                       matched_target_optimizer_steps=matched, **run.metrics_at(matched),
                       full_training_wall_s=run.final_wall)
            individual.append(row)
        per_seed.extend(individual)
        for metric in (*METRICS,'full_training_wall_s'):
            paired_values = []
            other = 'opd_reflex' if method == 'fastgrpo' else 'fastgrpo'
            for row in individual:
                run = by_method[other][row['seed']]
                other_value = run.final_wall if metric == 'full_training_wall_s' else run.metrics_at(matched)[metric]
                if row[metric] is not None and other_value is not None:
                    paired_values.append(row[metric])
            report.append(dict(method=method, metric=metric, matched_target_optimizer_steps=matched,
                **stat(paired_values)))
    for metric in SPEED_METRICS:
        speedups = []
        for seed in seeds:
            baseline = by_method['fastgrpo'][seed].metrics_at(matched)[metric]
            spark = by_method['opd_reflex'][seed].metrics_at(matched)[metric]
            ratio = baseline/spark if baseline is not None and spark else None
            speedups.append(ratio)
            per_seed.append(dict(seed=seed, method='spark_over_fastgrpo', matched_target_optimizer_steps=matched,
                                 metric=metric, speedup=ratio))
        report.append(dict(method='spark_over_fastgrpo', metric=metric, matched_target_optimizer_steps=matched,
                           **stat(speedups)))
    if ar_run is not None and matched in ar_run.updates:
        per_ar = ar_run.metrics_at(matched)
        for method in ('fastgrpo','opd_reflex'):
            for metric in SPEED_METRICS:
                values = [by_method[method][seed].metrics_at(matched)[metric] for seed in seeds]
                ratios = [per_ar[metric]/value for value in values if value and per_ar[metric] is not None]
                report.append(dict(method=f'{method}_over_fixed_ar', metric=metric,
                    matched_target_optimizer_steps=matched, **stat(ratios)))

    curves = []
    pair_map = {seed: [by_method[method][seed] for method in ('fastgrpo','opd_reflex')] for seed in seeds}
    accuracy_seeds = [seed for seed, pair in pair_map.items() if all(run.evaluations for run in pair)]
    time_seeds = [seed for seed, pair in pair_map.items() if all(run.final_wall is not None for run in pair)]
    accuracy_time_seeds = [seed for seed in accuracy_seeds if seed in time_seeds]
    time_runs = [run for seed in time_seeds for run in pair_map[seed]]
    accuracy_runs = [run for seed in accuracy_seeds for run in pair_map[seed]]
    budget_end = min((run.final_wall for run in time_runs), default=0.)
    eval_end = min((float(run.evaluations[-1][T]) for seed in accuracy_time_seeds for run in pair_map[seed]), default=0.)
    eval_end = min(budget_end,eval_end)
    budgets = sorted({budget_end*i/20 for i in range(21)} | ({eval_end} if eval_end else set()))
    eval_exact = sorted(set.intersection(*(set(int(row[U]) for row in run.evaluations) for run in accuracy_runs))) if accuracy_runs else []

    def aggregate_curve(axis, point, method, metric, values):
        curves.append(dict(axis=axis, point=point, method=method, metric=metric, **stat(values)))

    for method in ('fastgrpo','opd_reflex'):
        for index in exact:
            reward_seeds = [seed for seed,pair in pair_map.items()
                            if all(number(run.updates[index],'reward_mean') is not None for run in pair)]
            aggregate_curve('updates', index, method, 'reward',
                            [number(by_method[method][seed].updates[index],'reward_mean') for seed in reward_seeds])
        for index in eval_exact:
            aggregate_curve('updates', index, method, 'accuracy',
                            [by_method[method][seed].accuracy_at(index,'updates') for seed in accuracy_seeds])
        if eval_exact:
            eval_update = min(matched,eval_exact[-1])
            available = [value for value in eval_exact if value <= eval_update]
            if available:
                report.append(dict(method=method, metric='accuracy_latest_common_checkpoint',
                    matched_target_optimizer_steps=available[-1],
                    **stat(by_method[method][seed].accuracy_at(available[-1],'updates') for seed in accuracy_seeds)))
        timed_rows = {}
        for run in time_runs:
            observed = [(float(row[T]), row) for _,row in sorted(run.updates.items()) if number(row,T) is not None]
            timed_rows[run.path] = ([timestamp for timestamp,_ in observed], [row for _,row in observed])
        for budget in budgets:
            def reward_at(run):
                observations, rows = timed_rows[run.path]
                pos = bisect_right(observations,budget) - 1
                return number(rows[pos],'reward_mean') if pos >= 0 else None

            reward_seeds = [seed for seed in time_seeds if all(reward_at(run) is not None for run in pair_map[seed])]
            if reward_seeds:
                aggregate_curve('gpu_hours', budget/3600, method, 'reward',
                                [reward_at(by_method[method][seed]) for seed in reward_seeds])
            if budget <= eval_end and accuracy_time_seeds:
                observed = [seed for seed in accuracy_time_seeds
                            if all(run.accuracy_at(budget,'gpu_hours') is not None for run in pair_map[seed])]
                if observed:
                    aggregate_curve('gpu_hours', budget/3600, method, 'accuracy',
                                    [by_method[method][seed].accuracy_at(budget,'gpu_hours') for seed in observed])
    return dict(seeds=seeds, matched_target_optimizer_steps=matched,
                common_observed_budget_s=budget_end, common_accuracy_budget_s=eval_end, time_budget_seeds=time_seeds,
                accuracy_seeds=accuracy_seeds, summary=report, per_seed=per_seed, curves=curves,
                fixed_ar_reference=ar_run.path if ar_run else None)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plots(directory, cohort, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for axis, xlabel in (('updates','Target optimizer updates'), ('gpu_hours','Training GPU-hours')):
        for metric, ylabel in (('reward','Mean training reward'), ('accuracy','GSM8K accuracy')):
            fig, ax = plt.subplots(figsize=(5.4,3.1), layout='constrained')
            available = False
            for method,label,color in (('fastgrpo','FastGRPO','#3D5D86'), ('opd_reflex','SPARK','#AD4E38')):
                points = sorted((row for row in rows if row['axis']==axis and row['metric']==metric
                    and row['method']==method and row['mean'] is not None), key=lambda row: row['point'])
                if not points:
                    continue
                available = True
                x = [row['point'] for row in points]
                y = [row['mean'] for row in points]
                deviations = [float('nan') if row['sample_std'] is None else row['sample_std'] for row in points]
                ax.plot(x,y,label=label,color=color,marker='o',markersize=2.5,linewidth=1.5)
                ax.fill_between(x, [v-d for v,d in zip(y,deviations)],[v+d for v,d in zip(y,deviations)],
                                color=color,alpha=.17,linewidth=0)
            if available:
                ax.set(xlabel=xlabel,ylabel=ylabel)
                if metric == 'accuracy':
                    ax.set_ylim(0,1)
                ax.spines[['top','right']].set_visible(False)
                ax.grid(axis='y',alpha=.2)
                ax.legend(frameon=False)
                fig.savefig(directory / f'{cohort}_{metric}_by_{axis}.pdf')
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, action='append', default=[])
    parser.add_argument('--historical-manifest', type=Path)
    args=parser.parse_args()
    errors = []
    runs = []
    historical_rows = read_json(args.historical_manifest) if args.historical_manifest else []
    items = [(path,None) for path in (args.run_dir or sorted(args.runs_root.glob('*/summary.json')))]
    items = [(path.parent if path.name=='summary.json' else path, meta) for path,meta in items]
    if args.historical_manifest:
        items += [(args.historical_manifest.parent / row['run_dir'],row) for row in historical_rows]
    for path, historical in items:
        try:
            runs.append(Run.load(path,historical))
        except (OSError,ValueError,TypeError,KeyError) as failure:
            errors.append(dict(run_dir=str(path), reason=str(failure)))
    shared_groups = defaultdict(list)
    ar = []
    for run in runs:
        if run.method == 'ar':
            ar.append(run)
        elif run.method in ('fastgrpo','opd_reflex'):
            shared_groups[json.dumps(run.shared,sort_keys=True)].append(run)
    if len(ar) > 1:
        raise ValueError('provide at most one compatible fixed AR reference per invocation')
    cohorts = []
    for group in shared_groups.values():
        fast = [run for run in group if run.method == 'fastgrpo']
        variants = defaultdict(list)
        for run in group:
            if run.method == 'opd_reflex':
                variants[json.dumps(run.specific,sort_keys=True)].append(run)
        if not fast or not variants:
            errors.extend(dict(run_dir=run.path, reason='no other method with compatible shared configuration') for run in group)
            continue
        applicable_ar = ar[0] if ar and signature(fast[0].config,'fastgrpo',ar=True)[0] == ar[0].shared else None
        if ar and applicable_ar is None:
            errors.append(dict(run_dir=ar[0].path, reason='fixed AR reference is incompatible with the cohort'))
        for spark in variants.values():
            try:
                result = comparison(fast+spark,applicable_ar)
                paired_seeds = set(result['seeds'])
                errors.extend(dict(run_dir=run.path, reason='no compatible paired run for the same seed')
                              for run in fast+spark if run.seed not in paired_seeds)
            except ValueError as failure:
                errors.append(dict(run_dir=', '.join(run.path for run in fast+spark), reason=str(failure)))
            else:
                cohorts.append(dict(cohort=f'cohort_{len(cohorts)+1}', **result, compatibility=fast[0].shared,
                                    spark_configuration=spark[0].specific,
                                    entered_run_dirs=[run.path for run in fast+spark if run.seed in paired_seeds]))
    args.output.mkdir(parents=True)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(dict(cohorts=cohorts, excluded_runs=errors),stream,indent=2,allow_nan=False)
    for name in ('summary','per_seed','curves'):
        write_csv(args.output / f'{name}.csv', [dict(cohort=group['cohort'],**row)
                  for group in cohorts for row in group[name]])
    for group in cohorts:
        plots(args.output,group['cohort'],group['curves'])
    if not cohorts:
        raise SystemExit(f'No compatible paired runs. See {args.output / "summary.json"}')
    print(f'Wrote {len(cohorts)} paired cohort(s) to {args.output}')


if __name__=='__main__':
    main()
