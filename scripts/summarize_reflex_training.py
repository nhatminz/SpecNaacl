#!/usr/bin/env python3
"""Read-only comparison of two completed production training summaries."""
import argparse
import json
from pathlib import Path


def summarize(root):
    reports = {}
    throughput = {}
    for method in ('fastgrpo', 'specnaacl'):
        path = Path(root) / method / 'summary.json'
        if not path.is_file():
            raise FileNotFoundError(f'run incomplete; missing {path}')
        summary = json.loads(path.read_text(encoding='utf-8'))
        # Report the original authoritative counters/times, not invented
        # optimizer-steps/s from a GRPO batch index or accepted-token count.
        reports[method] = summary
        wall, generation = float(summary['total_wall_time_s']), float(summary['total_generate_time_s'])
        if wall <= 0 or generation <= 0:
            raise ValueError('completed benchmark must have positive wall/generation times')
        tokens, steps = int(summary['total_rollout_tokens']), int(summary['completed_grpo_steps'])
        throughput[method] = dict(end_to_end_tokens_per_s=tokens / wall,
            generation_tokens_per_s=tokens / generation, grpo_steps_per_s=steps / wall,
            wall_time_s=wall, generated_tokens=tokens, completed_grpo_steps=steps)
    a, b = throughput['fastgrpo'], throughput['specnaacl']
    matched_steps = a['completed_grpo_steps'] == b['completed_grpo_steps'] and a['completed_grpo_steps'] > 0
    return dict(benchmark='short_real_training', summaries=reports, throughput=throughput,
                matched_completed_steps=matched_steps,
                wall_speedup_at_matched_steps=a['wall_time_s'] / b['wall_time_s'] if matched_steps else None,
                note='Includes production startup/data/rollout/training overhead, no warmup exclusion. Compare matched steps/data/hardware and generation/training/total wall times, not AAL alone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
