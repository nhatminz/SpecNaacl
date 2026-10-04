#!/usr/bin/env bash
# Explicit short REAL GRPO comparison; never invoked by default training/tests.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_KEY:?Set MODEL_KEY to an existing configs/<model>/b200.env key}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-20}"
[[ "$BENCHMARK_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "BENCHMARK_STEPS must be positive" >&2; exit 2; }
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$REPO_DIR/outputs/benchmarks/reflex_training_$(date -u +%Y%m%dT%H%M%S_%N)}"
# Independent train roots isolate active_run/latest_run/checkpoint links too.
[[ ! -e "$BENCHMARK_ROOT" ]] || { echo "Use a NEW BENCHMARK_ROOT" >&2; exit 2; }
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}" RESUME="" STATISTICAL_TIME=False
export TARGET_LR="${TARGET_LR:-1e-5}" DRAFT_LR="${DRAFT_LR:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}" ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}" DATASET="${DATASET:-dapo}"
export REFLEX_PROFILE=0 REFLEX_DIAGNOSTICS=0
export PYTHON_BIN MODEL_KEY
for method in fastgrpo specnaacl; do
  env METHOD="$method" RUN_NAME="$method" RUN_DIR="$BENCHMARK_ROOT/$method" \
    TRAIN_MODEL_ROOT="$BENCHMARK_ROOT/${method}_links" \
    bash "$REPO_DIR/scripts/launch/train_model.sh" --max_grpo_steps "$BENCHMARK_STEPS" "$@"
done
if [[ "${DRY_RUN:-false}" == "true" ]]; then exit 0; fi
"$PYTHON_BIN" "$REPO_DIR/scripts/summarize_reflex_training.py" "$BENCHMARK_ROOT" \
  | tee "$BENCHMARK_ROOT/benchmark_report.json"
