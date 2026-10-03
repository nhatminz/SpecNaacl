import json
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.summarize_reflex_training import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_benchmark_smoke_verifies_parity_and_labels_scope():
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/benchmark_reflex.py"), "--device", "cpu",
        "--backend", "torch", "--batch", "2", "--vocab-size", "17",
        "--target-vocab-size", "23", "--hidden-size", "7", "--feature-dim", "3",
        "--draft-k", "3", "--draft-depth", "2", "--rounds", "3",
        "--warmup", "1", "--verify-rounds", "3",
    ], cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["backend"] == "torch"
    assert report["gpu_count_used"] == 0
    assert report["parity_passed"]
    assert report["max_probability_abs_error"] == report["max_state_abs_error"] == 0
    assert report["root_topk_id_agreement"] == 1
    assert report["candidate_ms_per_round"] > 0
    assert "CPU timings are smoke-only" in report["note"]


@pytest.mark.parametrize('scope', ['root', 'visited_path'])
def test_pipeline_benchmark_smoke_validates_actual_proposal_feedback_cycle(scope):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/benchmark_reflex_pipeline.py'),
        '--device', 'cpu', '--backend', 'torch', '--batch', '2', '--vocab', '17', '--target-vocab', '23',
        '--hidden', '7', '--contexts', '3', '--feature-dim', '4', '--topk', '3', '--depth', '2',
        '--warmup', '1', '--iterations', '2', '--feedback-scope', scope],
        cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['parity']['passed'] and report['gpu_name'] is None
    assert report['shape']['feedback_scope'] == scope
    timings = report['timings']['torch']
    assert {'correction_only', 'dense_correction_softmax_topk', 'fused_correction_topk',
            'visited_path_extraction', 'feedback_update', 'whole_reflex_verification_cycle'} == set(timings)
    assert all(item['wall_ms_median'] > 0 and item['gpu_ms_median'] is None for item in timings.values())


def test_real_rollout_benchmark_does_not_substitute_cpu_for_actual_measurement():
    from scripts.benchmark_reflex_rollout import parse_args, benchmark
    from unittest.mock import patch
    args = parse_args([flag for name in ('target-model', 'draft-config', 'draft-checkpoint', 'vocab-mapping', 'dataset-path')
                       for flag in ('--' + name, 'unused')])
    with patch('torch.cuda.is_available', return_value=False), pytest.raises(RuntimeError, match='requires CUDA'):
        benchmark(args)


def test_training_report_uses_correct_generation_and_end_to_end_denominators(tmp_path):
    for method, wall, gen in [('fastgrpo', 10, 6), ('specnaacl', 8, 4)]:
        run = tmp_path / method
        run.mkdir()
        (run / 'summary.json').write_text(json.dumps(dict(total_wall_time_s=wall, total_generate_time_s=gen,
            total_rollout_tokens=120, completed_grpo_steps=20)), encoding='utf-8')
    report = summarize(tmp_path)
    assert report['throughput']['fastgrpo']['end_to_end_tokens_per_s'] == 12
    assert report['throughput']['fastgrpo']['generation_tokens_per_s'] == 20
    assert report['throughput']['specnaacl']['generation_tokens_per_s'] == 30
    assert report['wall_speedup_at_matched_steps'] == 1.25
    assert report['matched_completed_steps']
