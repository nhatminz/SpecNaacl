import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
BASH = os.environ.get("BASH_BIN") or shutil.which("bash") or "bash"
PYTHON = os.environ.get("PYTHON") or Path(sys.executable).as_posix()
MODEL_KEYS = (
    "qwen25_1p5b", "qwen25_3b", "qwen25_7b", "qwen25_14b",
    "qwen3_1p7b", "qwen3_4b", "llama31_8b",
)


def test_all_launch_shells_have_valid_syntax_and_dry_run():
    env = dict(os.environ, DRY_RUN="true", PYTHON_BIN=PYTHON)
    scripts = [ROOT / f"train_{key}.sh" for key in MODEL_KEYS]
    scripts += [ROOT / f"pretrain_{key}.sh" for key in MODEL_KEYS]
    scripts += [
        ROOT / "scripts" / "run_fastgrpo_fair.sh",
        ROOT / "scripts" / "run_specnaacl.sh",
    ]
    scripts += [ROOT / "scripts" / "plot_training_time.sh"]
    for script in scripts + [ROOT / "pretrain_eagle3_sharegpt_b200.sh",
                             ROOT / "scripts/benchmark_pretrain.sh"]:
        subprocess.run([BASH, "-n", str(script)], check=True)
    for script in scripts[:-1]:
        result = subprocess.run(
            [BASH, str(script)], env=env, cwd=ROOT,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}\n{result.stdout}"


def test_fair_launchers_differ_only_by_method_and_reflex_toggle(tmp_path):
    env = dict(
        os.environ,
        DRY_RUN="true",
        PYTHON_BIN=PYTHON,
        RUN_NAME="fair-parity",
        RUN_DIR=str(tmp_path / "fair-parity"),
    )

    def command_for(name):
        result = subprocess.run(
            [BASH, str(ROOT / "scripts" / name)], env=env, cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        command_line = next(
            line[len("Command  :"):]
            for line in result.stdout.splitlines()
            if line.startswith("Command  :")
        )
        return shlex.split(command_line)

    def without_method_flags(command):
        normalized = list(command)
        for flag in ("--method", "--reflex_mode"):
            index = normalized.index(flag)
            del normalized[index:index + 2]
        return normalized

    baseline = command_for("run_fastgrpo_fair.sh")
    treatment = command_for("run_specnaacl.sh")
    assert baseline[baseline.index("--method") + 1] == "fastgrpo"
    assert baseline[baseline.index("--reflex_mode") + 1] == "off"
    assert treatment[treatment.index("--method") + 1] == "specnaacl"
    assert treatment[treatment.index("--reflex_mode") + 1] == "active"
    assert without_method_flags(baseline) == without_method_flags(treatment)


def test_pretrain_wrapper_reports_explicit_backend_topology_and_batch():
    env = dict(os.environ, DRY_RUN="true", PYTHON_BIN=PYTHON,
               PRETRAIN_ATTENTION_BACKEND="sdpa", PRETRAIN_DISTRIBUTED_MODE="auto",
               PRETRAIN_LENGTH_BUCKETING="true", PRETRAIN_DATALOADER_WORKERS="8",
               PRETRAIN_BATCH_SIZE="4", PRETRAIN_ACCUMULATION_STEPS="2",
               NPROC_PER_NODE="4", CUDA_VISIBLE_DEVICES="0,1,2,3")
    result = subprocess.run([BASH, str(ROOT / "pretrain_qwen25_3b.sh")], env=env,
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "attention=sdpa distributed=auto buckets=true workers=8 effective_batch=32" in result.stdout
    wrapper = (ROOT / "scripts/launch/pretrain_model.sh").read_text(encoding="utf-8")
    for name in ("PRETRAIN_ATTENTION_BACKEND", "PRETRAIN_DISTRIBUTED_MODE",
                 "PRETRAIN_LENGTH_BUCKETING", "PRETRAIN_LENGTH_BUCKET_BOUNDARIES",
                 "PRETRAIN_DATALOADER_WORKERS", "PRETRAIN_MAX_LENGTH",
                 "PRETRAIN_COMPACT_TEACHER", "PRETRAIN_OPTIMIZER_CPU_OFFLOAD"):
        assert f'{name}="${name}"' in wrapper
