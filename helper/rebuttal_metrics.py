"""Run metadata and buffered per-rollout/update telemetry for NAACL experiments."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def asset_id(path):
    """Hash the actual input files, with names relative to their resolved root."""
    root = Path(path).resolve(strict=True)
    if root.is_dir():
        files = [file for file in sorted(root.rglob('*')) if file.is_file() and not
                 any(part.startswith('.') for part in file.relative_to(root).parts)]
        checkpoint = root / 'draft.pth'
        if checkpoint.is_file():
            files = [checkpoint]
    else:
        files = [root]
    outer = hashlib.sha256()
    for file in files:
        part = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                part.update(chunk)
        if len(files) > 1:
            outer.update(file.relative_to(root).as_posix().encode() + b'\0')
        outer.update(part.digest())
    return {'path': str(path), 'resolved_path': str(root), 'sha256': outer.hexdigest()}


def runtime_environment(project, device):
    def command(*args):
        try:
            return subprocess.check_output(args, cwd=project, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    packages = {}
    for package in ('torch', 'transformers', 'peft', 'triton', 'datasets', 'numpy', 'pandas',
                    'pyarrow', 'accelerate', 'safetensors', 'tokenizers', 'math-verify',
                    'latex2sympy2-extended', 'sympy', 'packaging'):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    props = torch.cuda.get_device_properties(device) if torch.device(device).type == 'cuda' else None
    dirty = command('git', 'status', '--porcelain')
    return dict(python=platform.python_version(), software=packages, cuda_build=torch.version.cuda,
                torch_version=torch.__version__, gpu=props.name if props else None,
                gpu_memory_bytes=props.total_memory if props else None,
                gpu_compute_capability=[props.major, props.minor] if props else None,
                visible_cuda_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                driver=command('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
                gpu_count=torch.cuda.device_count(), world_size=int(os.getenv('WORLD_SIZE', '1')),
                parallelism='single_process_single_gpu', git_commit=command('git', 'rev-parse', 'HEAD'),
                git_dirty=bool(dirty) if dirty is not None else None)


class CSVLog:
    def __init__(self, path):
        self.stream = Path(path).open('x', newline='', buffering=65536)
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow({key: json.dumps(value, separators=(',', ':')) if
                              isinstance(value, (list, dict)) else value for key, value in row.items()})

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def length_metrics(lengths, responses, eos_token, legacy_cap, response_cap, prompts, repeats):
    values = np.asarray(lengths)
    average = float(values.mean())
    deviation = float(values.std())
    eos = [bool(row and row[-1] == eos_token) for row in responses]
    reached = [n >= (response_cap if response_cap is not None else legacy_cap - prompts[i // repeats])
               for i, n in enumerate(lengths)]
    percentiles = dict(zip(('length_p50', 'length_p90', 'length_p95', 'length_p99'),
                           map(float, np.percentile(values, [50, 90, 95, 99]))))
    return dict(generated_tokens=int(values.sum()), length_mean=average, length_std=deviation,
                length_min=int(values.min()), length_max=int(values.max()), length_cv=deviation/average if average else 0,
                fraction_over_2048=float((values > 2048).mean()), fraction_reaching_4096=float((values >= 4096).mean()),
                eos_rate=statistics.mean(eos), length_cap_rate=statistics.mean(
                    hit and not done for hit, done in zip(reached, eos)), **percentiles)


def reward_metrics(groups):
    samples = [sample for group, _ in groups for sample in group]
    eligible = [sample[0] for group, active in groups if active for sample in group]
    rewards = [sample[0] for sample in samples]
    return dict(reward_mean=statistics.mean(rewards), reward_std=statistics.pstdev(rewards),
                correctness_mean=statistics.mean(sample[1] for sample in samples),
                format_reward_mean=statistics.mean(sample[2] for sample in samples),
                eligible_reward_mean=statistics.mean(eligible) if eligible else None,
                eligible_prompt_groups=sum(active for _, active in groups),
                zero_variance_prompt_groups=sum(not active for _, active in groups),
                all_correct_prompt_groups=sum(all(sample[1] == 1 for sample in group) for group, _ in groups),
                all_incorrect_prompt_groups=sum(all(sample[1] == 0 for sample in group) for group, _ in groups))


def phase_seconds(ticket):
    name, host_s, start, end = ticket
    if end is None:
        return host_s
    end.synchronize()
    return start.elapsed_time(end) / 1000


class RebuttalRun:
    def __init__(self, args, run_dir, environment, inputs, dataset_info, initialization, started_at):
        self.args = args
        self.path = Path(run_dir)
        self.started_at = started_at
        self.process_start_timestamp = (datetime.now(timezone.utc) - timedelta(
            seconds=time.perf_counter()-started_at)).isoformat()
        self.io_s = 0.
        self.totals = defaultdict(float)
        self.since_update = defaultdict(float)
        self.identifiers = dict(experiment=args.rebuttal_experiment, model=args.rebuttal_model_key,
                                dataset=args.train_option, method=args.method, seed=args.seed)
        self.flush_interval = args.rollout_log_flush_interval
        logs = self.path / 'logs'
        self.rollouts = CSVLog(logs / 'rollout_metrics.csv')
        self.updates = CSVLog(logs / 'optimizer_updates.csv')
        self.profiles = CSVLog(logs / 'generation_profile.csv')
        self.memory = CSVLog(logs / 'memory_profile.csv')
        self.rounds = CSVLog(logs / 'verification_rounds.csv') if args.generation_profile == 'detailed' and args.round_log_frequency else None
        self.responses = (logs / 'response_metrics.jsonl').open('x', buffering=65536)
        self.streams = [stream for stream in (self.rollouts, self.updates, self.profiles, self.memory,
                                              self.rounds, self.responses) if stream is not None]
        snapshot_dir = self.path / 'checkpoints/evaluation'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = snapshot_dir
        json_write(self.path / 'config_resolved.yaml', dict(
            identifiers=self.identifiers, args=vars(args), inputs=inputs, dataset=dataset_info,
            initialization=initialization, environment=environment, metric_protocol='rebuttal_v1',
            proposal_profile_sha256=(asset_id(os.environ['OPD_PROPOSAL_PROFILE'])['sha256']
                                     if os.getenv('OPD_PROPOSAL_PROFILE') and args.method == 'opd_reflex' else None),
            proposal_env={key: value for key, value in os.environ.items()
                          if key.startswith('OPD_') or key == 'KV_GATHER_STRATEGY'},
            historical_reference=os.getenv('REBUTTAL_HISTORICAL_REFERENCE', 'unverified_launcher_defaults'),
            generation_timing_basis='perf_counter_through_generation_stream_boundary',
            seeds=dict(training=args.seed, dataset=args.train_subset_seed, dataloader=args.prompt_order_seed,
                       projector=getattr(args,'opd_projector_seed',None),
                       reflex=getattr(args,'reflex_seed',None), evaluation=args.evaluation_seed)))
        self.training_start = time.perf_counter()
        self.start_timestamp = utc_now()

    def wall(self):
        return time.perf_counter() - self.training_start - self.io_s

    def timed_io(self, function, *args, **kwargs):
        begin = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.io_s += time.perf_counter() - begin

    def snapshot(self, model, updates, kind, wall=None):
        now = (0. if kind == 'initial' else self.wall()) if wall is None else wall
        path = self.snapshot_dir / f'{kind}_updates_{updates:07d}'
        path.mkdir(exist_ok=False)
        before = self.io_s
        self.timed_io(model.save_pretrained, path)
        json_write(path / 'snapshot.json', dict(**self.identifiers, kind=kind, target_optimizer_steps=updates,
                   cumulative_training_wall_time_s=now, checkpoint_io_s=self.io_s - before))

    def observe(self, outputs, groups, prompts, lengths, eos_token, draft_s):
        self.totals['rollouts'] += 1
        self.totals['prompts'] += len(prompts)
        self.totals['draft_s'] += draft_s
        self.totals['generate_s'] += outputs['total_time_cost']
        self.totals['tokens'] += sum(lengths)
        for name in ('total_acc_length', 'total_decoded_token_num', 'total_accepted_draft_tokens',
                     'total_proposed_draft_tokens', 'batch_verification_rounds', 'verified_tree_nodes'):
            self.totals[name] += outputs[name]
        self.since_update['rollouts'] += 1
        self.since_update['prompts'] += len(prompts)
        for group, _ in groups:
            for reward, correct, _ in group:
                self.totals['rewards'] += reward
                self.totals['correct'] += correct
                self.totals['responses'] += 1
                self.since_update['rewards'] += reward
                self.since_update['correct'] += correct
                self.since_update['responses'] += 1
        index = int(self.totals['rollouts'])
        profile = outputs['generation_profile']
        common = dict(self.identifiers, rollout_index=index)
        repeats = self.args.repeated_generate_nums
        responses = outputs['generated_token_ids']
        stats = length_metrics(lengths, responses, eos_token, self.args.max_length,
                               self.args.max_response_tokens, prompts, repeats)
        for group_id, (group, active) in enumerate(groups):
            for response_id, (reward, correct, format_reward) in enumerate(group):
                row = group_id * repeats + response_id
                self.responses.write(json.dumps(dict(**common, group=group_id, response=response_id,
                    prompt_tokens=prompts[group_id], generated_tokens=lengths[row], total_reward=reward,
                    correctness_reward=correct, format_reward=format_reward, eligible=active,
                    eos=bool(responses[row] and responses[row][-1] == eos_token)))+'\n')
        self.profiles.write(dict(**common, opd_backend=outputs.get('opd_backend','off'),
                                 proposal_profile=outputs.get('opd_proposal_profile_effective'), **profile))
        self.memory.write(dict(**common, initial_concurrency=len(responses), active_prompt_groups=len(prompts),
                               **outputs['generation_memory'],
                               spark_correction_application_time_ms=profile['spark_correction_application_time_ms'],
                               spark_correction_update_time_ms=profile['spark_correction_update_time_ms'],
                               spark_updates=outputs.get('opd_updates', 0),
                               fastlk_updates=outputs.get('reflex_updates', 0)))
        if self.rounds:
            for row in outputs['verification_rounds']:
                self.rounds.write(dict(**common, **row))
        rounds = outputs['total_decoded_token_num']
        return dict(**common, prompt_count=len(prompts), response_count=len(responses),
                    initial_concurrency=len(responses), active_prompt_groups=len(prompts),
                    prompt_lengths=prompts, context_lengths=[prompts[i // repeats]+n for i,n in enumerate(lengths)],
                    **stats, **reward_metrics(groups), aal=outputs['total_acc_length']/max(1, rounds),
                    acceptance_rate=outputs['draft_acceptance_rate'], active_response_rounds=outputs['active_response_rounds'],
                    active_prompt_group_rounds=outputs['active_prompt_group_rounds'],
                    verification_batches=outputs['batch_verification_rounds'], response_verification_rounds=rounds,
                    verified_tree_nodes=outputs['verified_tree_nodes'],
                    accepted_draft_tokens=outputs['total_accepted_draft_tokens'],
                    proposed_draft_tokens=outputs['total_proposed_draft_tokens'],
                    spark_feedback_states=outputs.get('opd_selected_states', 0), spark_updates=outputs.get('opd_updates', 0),
                    generation_wall_time_s=outputs['total_time_cost'], throughput_tokens_s=sum(lengths)/outputs['total_time_cost'],
                    draft_training_time_s=draft_s)

    def cumulative(self):
        t = self.totals
        return dict(cumulative_generated_tokens=int(t['tokens']), cumulative_responses=int(t['responses']),
                    cumulative_reward_mean=t['rewards']/t['responses'] if t['responses'] else None,
                    cumulative_correctness_mean=t['correct']/t['responses'] if t['responses'] else None,
                    cumulative_generation_time_s=t['generate_s'],
                    cumulative_draft_training_time_s=t['draft_s'], cumulative_target_training_time_s=t['target_s'],
                    cumulative_training_wall_time_s=self.wall(), cumulative_checkpoint_io_s=self.io_s,
                    cumulative_aal=t['total_acc_length']/max(1, t['total_decoded_token_num']),
                    cumulative_acceptance_rate=t['total_accepted_draft_tokens']/max(1,t['total_proposed_draft_tokens']),
                    cumulative_generation_tokens_s=t['tokens']/max(1e-12,t['generate_s']))

    def optimizer_update(self, updates, draft_updates, target_s, target_loss, draft_losses, target_opt, draft_opt,
                         eligible_reward):
        self.totals['target_s'] += target_s
        delta = self.since_update
        self.updates.write(dict(**self.identifiers, target_optimizer_steps=updates, draft_optimizer_steps=draft_updates,
            rollout_index=int(self.totals['rollouts']), rollouts_since_update=int(delta['rollouts']),
            prompts_since_update=int(delta['prompts']), target_loss=target_loss,
            draft_feature_loss=draft_losses[0], draft_distribution_loss=draft_losses[1],
            target_learning_rate=target_opt.param_groups[0]['lr'],
            draft_learning_rates=[group['lr'] for group in draft_opt.param_groups],
            reward_mean=delta['rewards']/delta['responses'] if delta['responses'] else None,
            correctness_mean=delta['correct']/delta['responses'] if delta['responses'] else None,
            eligible_reward_mean=eligible_reward, **self.cumulative()))
        delta.clear()

    def finish_rollout(self, row, target_updates, draft_updates, target_s, started):
        self.rollouts.write(dict(**row, target_optimizer_steps=target_updates, draft_optimizer_steps=draft_updates,
            target_training_time_s=target_s, iteration_wall_time_s=self.wall()-started,
            cumulative_training_wall_time_s=self.wall(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        for kind, value in (('allocated', torch.cuda.max_memory_allocated()), ('reserved', torch.cuda.max_memory_reserved())):
            key = f'peak_{kind}_bytes'
            self.totals[key] = max(self.totals[key], value)
        if row['rollout_index'] % self.flush_interval == 0:
            self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def finish(self, training_wall_time):
        for stream in self.streams:
            stream.close()
        t = self.totals
        cumulative = self.cumulative()
        cumulative['cumulative_training_wall_time_s'] = training_wall_time
        return dict(**self.identifiers, **cumulative, training_wall_time_s=training_wall_time,
                    peak_allocated_bytes=int(t['peak_allocated_bytes']), peak_reserved_bytes=int(t['peak_reserved_bytes']),
                    setup_time_s=self.training_start-self.started_at, checkpoint_io_s=self.io_s,
                    evaluation_time_s=None, started_at=self.process_start_timestamp,
                    training_started_at=self.start_timestamp, finished_at=utc_now(),
                    actual_rollouts=int(t['rollouts']), actual_prompts=int(t['prompts']),
                    reward_mean=t['rewards']/max(1,t['responses']), correctness_mean=t['correct']/max(1,t['responses']),
                    average_response_length=t['tokens']/max(1,t['responses']),
                    absolute_process_wall_time_s=time.perf_counter()-self.started_at)
