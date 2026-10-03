import json
from pathlib import Path
import subprocess
import sys


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
