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

[[ -z "${RUN_DIR:-}${RUN_NAME:-}${RESUME:-}" ]] ||
  fail "rebuttal starts in a fresh uniquely named directory. Unset RUN_DIR, RUN_NAME and RESUME"
if [[ "$REBUTTAL_EXPERIMENT" == multiseed ]]; then
  if [[ -n "${HISTORICAL_CONFIG:-}" && -f "$HISTORICAL_CONFIG" ]]; then
    # Explicit operator-provided experiment settings, captured from seed 42.
    source "$HISTORICAL_CONFIG"
    export REBUTTAL_HISTORICAL_REFERENCE="$HISTORICAL_CONFIG"
  elif [[ -n "${HISTORICAL_CONFIG:-}" && "${DRY_RUN:-false}" != true ]]; then
    fail "historical configuration file missing: $HISTORICAL_CONFIG"
  else
    printf 'Seed-42 manifests are unavailable. Multi-seed settings are unverified against the historical run (B scope: %s).\n' "${OPD_B_SHARING_SCOPE:-rollout_batch}" >&2
    export REBUTTAL_HISTORICAL_REFERENCE=unverified_launcher_defaults
  fi
  export OPD_B_SHARING_SCOPE="${OPD_B_SHARING_SCOPE:-rollout_batch}"
  export OPD_B_RESET_MODE="${OPD_B_RESET_MODE:-rollout}"
  export OPD_UPDATE_INTERVAL_ROUNDS="${OPD_UPDATE_INTERVAL_ROUNDS:-1}"
  export OPD_TRAIN_PROJECTOR="${OPD_TRAIN_PROJECTOR:-1}"
else
  export OPD_B_SHARING_SCOPE=per_prompt_group OPD_B_RESET_MODE=rollout OPD_UPDATE_INTERVAL_ROUNDS=1 OPD_TRAIN_PROJECTOR=1
fi

export SEED="$training_seed" BATCH_SIZE=8 RESPONSES_PER_PROMPT="$response_count"
export MAX_RESPONSE_TOKENS="$response_cap"
export TRAIN_SUBSET_SEED=42
export PROMPT_ORDER_SEED="${PROMPT_ORDER_SEED:-42}"
export EVALUATION_SEED="${EVALUATION_SEED:-42}"
export EVAL_INTERVAL_UPDATES="${EVAL_INTERVAL_UPDATES:-50}"
export EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${GSM8K_TEST:-$DATA_ROOT/gsm8k/main/test-00000-of-00001.parquet}}"
export PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-${PRETRAIN_OUTPUT_ROOT:-$PROJECT_DIR/outputs}/pretrain/$MODEL_KEY}"
export DRAFT_INITIALIZATION_MODE=pretrained
export OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$PROJECT_DIR/outputs/benchmarks/opd_proposals}"
export TRAIN_MODEL_ROOT="$REBUTTAL_OUTPUT_ROOT/$REBUTTAL_EXPERIMENT"
if [[ "$REBUTTAL_EXPERIMENT" == concurrency ]]; then
  export GENERATION_PROFILE="${GENERATION_PROFILE:-detailed}"
else
  export GENERATION_PROFILE="${GENERATION_PROFILE:-clean}"
fi
export ROUND_LOG_FREQUENCY="${ROUND_LOG_FREQUENCY:-20}"
export ROLLOUT_LOG_FLUSH_INTERVAL="${ROLLOUT_LOG_FLUSH_INTERVAL:-10}"
export STATISTICAL_TIME=False OPD_PROFILE=0 OPD_DIAGNOSTICS=0
one_gpu
source "$PROJECT_DIR/scripts/launch/train_model.sh"
