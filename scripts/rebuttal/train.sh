#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export REBUTTAL_EXPERIMENT="$1" MODEL_KEY="$2" METHOD="$3"
training_seed="$4"
response_count="$5"
response_cap="$6"
shift 6
one_gpu
dataset_defaults

if [[ "$REBUTTAL_EXPERIMENT" == multiseed && -n "${HISTORICAL_CONFIG:-}" ]]; then
  if [[ -f "$HISTORICAL_CONFIG" ]]; then
    selected_method="$METHOD"
    source "$HISTORICAL_CONFIG"
    export METHOD="$selected_method" REBUTTAL_HISTORICAL_REFERENCE="$HISTORICAL_CONFIG"
  elif [[ "${DRY_RUN:-false}" != true ]]; then
    fail "historical configuration file missing: $HISTORICAL_CONFIG"
  fi
fi

export SEED="$training_seed" BATCH_SIZE=8 RESPONSES_PER_PROMPT="$response_count"
export MAX_RESPONSE_TOKENS="$response_cap"
export TRAIN_SUBSET_SEED=42
export PROMPT_ORDER_SEED="${PROMPT_ORDER_SEED:-42}"
export EVALUATION_SEED="${EVALUATION_SEED:-42}"
export EVAL_INTERVAL_UPDATES="${EVAL_INTERVAL_UPDATES:-50}"
export EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${GSM8K_TEST:-$DATA_ROOT/gsm8k/main/test-00000-of-00001.parquet}}"
export PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-${PRETRAIN_OUTPUT_ROOT:-$PROJECT_DIR/outputs}/pretrain/$MODEL_KEY}"
if [[ "$REBUTTAL_EXPERIMENT" == concurrency ]]; then
  export GENERATION_PROFILE="${GENERATION_PROFILE:-detailed}"
else
  export GENERATION_PROFILE="${GENERATION_PROFILE:-clean}"
fi
export ROUND_LOG_FREQUENCY="${ROUND_LOG_FREQUENCY:-20}"
export ROLLOUT_LOG_FLUSH_INTERVAL="${ROLLOUT_LOG_FLUSH_INTERVAL:-10}"
exec bash "$PROJECT_DIR/scripts/launch/train_specnaacl.sh"
