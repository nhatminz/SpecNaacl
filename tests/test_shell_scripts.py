import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_KEYS = (
    "qwen25_1p5b", "qwen25_3b", "qwen25_7b", "qwen25_14b",
    "qwen3_1p7b", "qwen3_4b", "llama31_8b",
)


def test_all_launch_shells_have_valid_syntax_and_dry_run():
    env = dict(os.environ, DRY_RUN="true", PYTHON_BIN=os.environ.get("PYTHON", "python3"))
    scripts = [ROOT / f"train_{key}.sh" for key in MODEL_KEYS]
    scripts += [ROOT / f"pretrain_{key}.sh" for key in MODEL_KEYS]
    scripts += [ROOT / "scripts" / "plot_training_time.sh"]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    for script in scripts[:-1]:
        result = subprocess.run(
            ["bash", str(script)], env=env, cwd=ROOT,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}\n{result.stdout}"
