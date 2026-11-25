#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
one_gpu
run="${1:-${RUN_DIR:-}}"
if [[ -z "$run" ]]; then
  [[ "${DRY_RUN:-false}" == true ]] || fail 'provide a training run directory as the first argument or RUN_DIR'
  run="$REBUTTAL_OUTPUT_ROOT/<training-run>"
fi
eval_data="${EVAL_DATASET_PATH:-${GSM8K_TEST:-$DATA_ROOT/gsm8k/main/test-00000-of-00001.parquet}}"
output="${EVALUATION_OUTPUT_DIR:-$run/logs}"
cmd=("$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_rebuttal.py" --run-dir "$run" --output "$output"
  --dataset-path "$eval_data" --batch-size "${EVAL_BATCH_SIZE:-8}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-2048}"
  --max-samples "${EVAL_MAX_SAMPLES:-0}" --temperature "${EVAL_TEMPERATURE:-1.0}" --top-p "${EVAL_TOP_P:-0.95}")
[[ -z "${EVALUATION_SEED:-}" ]] || cmd+=(--seed "$EVALUATION_SEED")
[[ -z "${TARGET_MODEL:-}" ]] || cmd+=(--target-model "$TARGET_MODEL")
[[ -z "${EVAL_CHECKPOINT_MANIFEST:-}" ]] || cmd+=(--checkpoint-manifest "$EVAL_CHECKPOINT_MANIFEST")
[[ -z "${EVAL_MAX_PROMPT_LENGTH:-}" ]] || cmd+=(--max-prompt-length "$EVAL_MAX_PROMPT_LENGTH")
[[ -z "${EVAL_MODEL_DTYPE:-}" ]] || cmd+=(--dtype "$EVAL_MODEL_DTYPE")
[[ -z "${EVAL_ATTENTION_IMPLEMENTATION:-}" ]] || cmd+=(--attention-implementation "$EVAL_ATTENTION_IMPLEMENTATION")
[[ "${EVAL_SAMPLE:-false}" != true ]] || cmd+=(--sample)
show_command env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
[[ "${DRY_RUN:-false}" == true ]] && exit 0
[[ -d "$run" ]] || fail "training run does not exist: $run"
[[ -e "$eval_data" ]] || fail "held-out dataset missing, set GSM8K_TEST: $eval_data"
[[ -z "${EVAL_CHECKPOINT_MANIFEST:-}" || -f "$EVAL_CHECKPOINT_MANIFEST" ]] || fail "missing checkpoint manifest: $EVAL_CHECKPOINT_MANIFEST"
[[ -n "${EVAL_CHECKPOINT_MANIFEST:-}" || -f "$run/config_resolved.yaml" ]] || fail "missing resolved training config under $run"
[[ ! -e "$output/evaluation_metrics.csv" ]] || fail "set a fresh EVALUATION_OUTPUT_DIR, evaluation already exists: $output"
mkdir -p "$output"
"${cmd[@]}" 2>&1 | tee "$output/evaluation_console.log"
