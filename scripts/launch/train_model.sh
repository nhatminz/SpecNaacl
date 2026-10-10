#!/usr/bin/env bash
# Generic SpecNaacl launcher; sourced by the thin per-model wrappers.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_ENV="${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
MODEL_ENV="${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
source "$COMMON_ENV"
source "$MODEL_ENV"
# New optional settings must not rely on all server wrappers/configs having been
# updated together. Keep defaults here too, before ANY expansion under set -u.
# An explicit environment/config override always takes precedence.
export ROLLOUT_LOG_FLUSH_INTERVAL="${ROLLOUT_LOG_FLUSH_INTERVAL:-1}"
export OPD_PROJECTOR_LR="${OPD_PROJECTOR_LR:-}"
export OPD_PROJECTOR_INIT="${OPD_PROJECTOR_INIT:-head_aligned}"
export OPD_PROJECTOR_SEED="${OPD_PROJECTOR_SEED:-42}"
export OPD_UPDATE_INTERVAL_ROUNDS="${OPD_UPDATE_INTERVAL_ROUNDS:-1}"
export OPD_ABLATION_NAME="${OPD_ABLATION_NAME:-}"
export OPD_PROPOSAL_PROFILE="${OPD_PROPOSAL_PROFILE:-}"
export OPD_PROPOSAL_MODE="${OPD_PROPOSAL_MODE:-auto}"
export OPD_DENSE_IMPLEMENTATION="${OPD_DENSE_IMPLEMENTATION:-auto}"
export OPD_AUTO_TUNE_IF_MISSING="${OPD_AUTO_TUNE_IF_MISSING:-0}"
export OPD_UPDATE_STREAM="${OPD_UPDATE_STREAM:-1}"
export OPD_FAST_LR="${OPD_FAST_LR:-${FAST_LR:-0.01}}"
export OPD_KV_MAX_RETAINED_TOKENS="${OPD_KV_MAX_RETAINED_TOKENS:-0}"
if [[ "$METHOD" == opd_reflex ]]; then
  export OPD_SAMPLER_MODE="${OPD_SAMPLER_MODE:-finite}"
fi
if [[ ! "$ROLLOUT_LOG_FLUSH_INTERVAL" =~ ^0*[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLOUT_LOG_FLUSH_INTERVAL must be a positive integer; got: $ROLLOUT_LOG_FLUSH_INTERVAL" >&2
  exit 2
fi
if [[ ! "$OPD_KV_MAX_RETAINED_TOKENS" =~ ^[0-9]+$ ]];then
  echo "ERROR: OPD_KV_MAX_RETAINED_TOKENS must be a nonnegative integer" >&2;exit 2
fi
: "${MODEL:?MODEL is required}"
case "$METHOD" in
  fastgrpo|opd_reflex) ;;
  *) echo "ERROR: METHOD must be fastgrpo or opd_reflex, got: $METHOD" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
case "${DATASET,,}" in
  gsm8k)
    TRAIN_OPTION="gsm8k"
    DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}"
    ;;
  simplelr|simplelr_abel|simplelr_abel_level3to5)
    TRAIN_OPTION="simplelr_abel_level3to5"
    DATASET="simplelr"
    DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}"
    ;;
  dapo|dapo-math|dapo_math)
    TRAIN_OPTION="DAPO-math"
    DATASET="dapo"
    DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}"
    ;;
  *)
    [[ -n "$DATASET_PATH" ]] || { echo "ERROR: unknown DATASET=$DATASET; set DATASET_PATH" >&2; exit 2; }
    TRAIN_OPTION="$DATASET"
    ;;
esac

PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$PRETRAIN_MODEL_ROOT/latest_checkpoint}"
TARGET_CONFIG="$MODEL/config.json"
DRAFT_INITIALIZATION_MODE="${DRAFT_INITIALIZATION_MODE:-pretrained}"
export OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$OUTPUT_ROOT/benchmarks/opd_proposals}"

TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-$OUTPUT_ROOT/train/$MODEL_KEY}"
# Each ablation owns its active/latest pointers. Parallel configurations never
# discover or overwrite another configuration's auto-resume run.
if [[ "$METHOD" == opd_reflex ]]; then
  case "$OPD_PROJECTOR_INIT" in head_aligned|random_orthogonal) ;; *) echo 'ERROR: invalid OPD_PROJECTOR_INIT' >&2;exit 2;; esac
  case "$OPD_UPDATE_INTERVAL_ROUNDS" in 0|1|5|10|15) ;; *) echo 'ERROR: invalid OPD_UPDATE_INTERVAL_ROUNDS' >&2;exit 2;; esac
  case "$OPD_TRAIN_PROJECTOR" in 0|1) ;; *) echo 'ERROR: invalid OPD_TRAIN_PROJECTOR' >&2;exit 2;; esac
  if [[ -z "$OPD_ABLATION_NAME" ]];then
    if [[ "$OPD_PROJECTOR_INIT" == random_orthogonal ]];then
      if [[ "$OPD_TRAIN_PROJECTOR" == 1 ]];then OPD_ABLATION_NAME=learned_random;else OPD_ABLATION_NAME=frozen_random;fi
    elif [[ "$OPD_TRAIN_PROJECTOR" == 0 ]];then OPD_ABLATION_NAME=frozen_head_aligned
    else OPD_ABLATION_NAME=main;fi
    if [[ "$OPD_UPDATE_INTERVAL_ROUNDS" != 1 ]];then OPD_ABLATION_NAME+="_interval$OPD_UPDATE_INTERVAL_ROUNDS";fi
  fi
  [[ "$OPD_ABLATION_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid ablation name' >&2;exit 2; }
  if [[ "$OPD_UPDATE_INTERVAL_ROUNDS" == 0 ]];then OPD_PROJECTOR_MODE=inactive
  elif [[ "$OPD_TRAIN_PROJECTOR" == 1 ]];then OPD_PROJECTOR_MODE=learned
  else OPD_PROJECTOR_MODE=frozen;fi
  if [[ "$OPD_ABLATION_NAME" != main || "$OPD_PROJECTOR_INIT" != head_aligned || "$OPD_TRAIN_PROJECTOR" != 1 || "$OPD_UPDATE_INTERVAL_ROUNDS" != 1 || "$OPD_PROJECTOR_SEED" != 42 ]];then
    TRAIN_MODEL_ROOT="$TRAIN_MODEL_ROOT/ablations/${OPD_ABLATION_NAME}__init-${OPD_PROJECTOR_INIT}__seed${OPD_PROJECTOR_SEED}__train${OPD_TRAIN_PROJECTOR}__interval${OPD_UPDATE_INTERVAL_ROUNDS}"
  fi
fi
REQUESTED_RUN_DIR="${RUN_DIR:-}"
timestamp="$(date -u +%Y%m%dT%H%M%S)"
short_uuid="$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${DATASET}__method-${METHOD}__seed${TRAIN_SUBSET_SEED}__${timestamp}__${short_uuid}}"
RUN_DIR="${RUN_DIR:-$TRAIN_MODEL_ROOT/$RUN_NAME}"
if [[ "$RESUME" == "auto" && -z "$REQUESTED_RUN_DIR" && -e "$TRAIN_MODEL_ROOT/active_run" ]]; then
  active_run="$(readlink -f "$TRAIN_MODEL_ROOT/active_run")"
  if [[ "$(basename "$active_run")" == *"method-${METHOD}"* && -f "$active_run/checkpoints/resume/latest.pt" ]]; then
    if "$PYTHON_BIN" - "$active_run/summary.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
sys.exit(0 if not path.exists() or json.loads(path.read_text()).get('stopped_by_max_grpo_steps', False) else 1)
PY
    then
      RUN_DIR="$active_run"
      RUN_NAME="$(basename "$RUN_DIR")"
    fi
  fi
fi
LOG_DIR="$RUN_DIR/logs"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

if [[ "$RESUME" == "auto" ]]; then
  RESUME_CHECKPOINT="$CHECKPOINT_DIR/resume/latest.pt"
  [[ -f "$RESUME_CHECKPOINT" ]] || RESUME_CHECKPOINT=""
elif [[ -n "$RESUME" ]]; then
  RESUME_CHECKPOINT="$RESUME"
else
  RESUME_CHECKPOINT=""
fi

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE"
  "$PROJECT_DIR/grpo_speculative.py"
  --method "$METHOD"
  --model_dir "$MODEL"
  --adapter_path "$DRAFT_CHECKPOINT"
  --draft_initialization_mode "$DRAFT_INITIALIZATION_MODE"
  --dtype "$MODEL_DTYPE"
  --attn_implementation "$ATTENTION_IMPLEMENTATION"
  --load_lora_path "$TARGET_ADAPTER"
  --model_type "$MODEL_TYPE"
  --train_option "$TRAIN_OPTION"
  --dataset_path "$DATASET_PATH"
  --train_data_fraction "$TRAIN_DATA_FRACTION"
  --train_subset_seed "$TRAIN_SUBSET_SEED"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --version_name "$RUN_NAME"
  --batch_size "$BATCH_SIZE"
  --num_epochs "$NUM_EPOCHS"
  --sample_num "$SAMPLE_NUM"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --draft_accumulation_steps "$DRAFT_ACCUMULATION_STEPS"
  --target_lr "$TARGET_LR"
  --draft_lr "$DRAFT_LR"
  --draft_train_profile "$DRAFT_TRAIN_PROFILE"
  --is_train_draft "$IS_TRAIN_DRAFT"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_length "$GEN_MAX_LENGTH"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_training_padding_gap "$MAX_TRAINING_PADDING_GAP"
  --max_training_token "$MAX_TRAINING_TOKEN"
  --logps_chunk_size "$LOGPS_CHUNK_SIZE"
  --grpo_iteration_num "$GRPO_ITERATION_NUM"
  --repeated_generate_nums "$RESPONSES_PER_PROMPT"
  --beta "$BETA"
  --epsilon "$EPSILON"
  --verification_capacity "$VERIFICATION_CAPACITY"
  --max_draft_token_length "$MAX_DRAFT_TOKEN_LENGTH"
  --max_draft_k "$MAX_DRAFT_K"
  --max_verification_num "$MAX_VERIFICATION_NUM"
  --min_draft_token_length "$MIN_DRAFT_TOKEN_LENGTH"
  --draft_token_length_c "$DRAFT_TOKEN_LENGTH_C"
  --statistical_time "$STATISTICAL_TIME"
  --num_workers "$NUM_WORKERS"
  --persistent_workers "$PERSISTENT_WORKERS"
  --log_interval "$LOG_INTERVAL"
  --rollout_log_flush_interval "$ROLLOUT_LOG_FLUSH_INTERVAL"
  --log_file "$LOG_DIR/metrics.jsonl"
  --timing_file "$LOG_DIR/timing.csv"
  --summary_file "$RUN_DIR/summary.json"
  --saved_model_dir "$CHECKPOINT_DIR/target"
  --saved_draft_model_dir "$CHECKPOINT_DIR/draft"
  --saved_statistics_dir "$RUN_DIR/statistics"
  --checkpoint_dir "$CHECKPOINT_DIR/resume"
  --save_checkpoint_steps "$SAVE_CHECKPOINT_STEPS"
  --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS"
  --resume_checkpoint "$RESUME_CHECKPOINT"
  --seed "$TRAIN_SUBSET_SEED"
)
if [[ "$METHOD" == opd_reflex ]]; then
  cmd+=(
  --opd_rank "$OPD_RANK" --opd_topk "$OPD_TOPK"
  --opd_fast_lr "$OPD_FAST_LR"
  --opd_visited_weight "$OPD_VISITED_WEIGHT" --opd_frontier_weight "$OPD_FRONTIER_WEIGHT"
  --opd_update_stream "$OPD_UPDATE_STREAM" --opd_backend "$OPD_BACKEND"
  --opd_profile "$OPD_PROFILE" --opd_diagnostics "$OPD_DIAGNOSTICS"
  --opd_train_projector "$OPD_TRAIN_PROJECTOR"
  --opd_projector_init "$OPD_PROJECTOR_INIT" --opd_projector_seed "$OPD_PROJECTOR_SEED"
  --opd_update_interval_rounds "$OPD_UPDATE_INTERVAL_ROUNDS" --opd_ablation_name "$OPD_ABLATION_NAME"
  --kv_gather_strategy "${KV_GATHER_STRATEGY:-stacked}"
  )
fi
if [[ -n "$OPD_PROJECTOR_LR" && "$METHOD" == opd_reflex ]]; then
  cmd+=(--opd_projector_lr "$OPD_PROJECTOR_LR")
fi
cmd+=(--max_target_optimizer_steps "${MAX_TARGET_OPTIMIZER_STEPS:-0}" --max_rollout_prompts "${MAX_ROLLOUT_PROMPTS:-0}")
if (($#)); then cmd+=("$@"); fi

printf 'Run name : %s\nRun dir  : %s\nModel    : %s\nDataset  : %s\nDraft    : %s\nMethod   : %s\nEngine   : %s\nGPUs     : %s\n' \
  "$RUN_NAME" "$RUN_DIR" "$MODEL" "$DATASET_PATH" "$DRAFT_CHECKPOINT" "$METHOD" "$METHOD" "$NPROC_PER_NODE"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-false}" == "true" ]]; then return 0 2>/dev/null || exit 0; fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --python-only
"$PYTHON_BIN" "$PROJECT_DIR/scripts/check_training_sources.py" --backend fastgrpo

[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$DATASET_PATH" ]] || { echo "ERROR: dataset not found: $DATASET_PATH" >&2; exit 2; }
if [[ "$DRAFT_INITIALIZATION_MODE" == "pretrained" && ! -e "$DRAFT_CHECKPOINT" ]]; then
  echo "ERROR: draft checkpoint not found: $DRAFT_CHECKPOINT" >&2; exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "ERROR: resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 2
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" \
  --requirements "$PROJECT_DIR/requirements.txt" --require-cuda

if [[ "$METHOD" == opd_reflex && "$OPD_PROPOSAL_MODE" == auto && "$DRAFT_INITIALIZATION_MODE" == pretrained ]];then
  OPD_PROPOSAL_PROFILE="$("$PYTHON_BIN" "$PROJECT_DIR/scripts/resolve_opd_profile.py" \
    --target-config "$TARGET_CONFIG" --draft-checkpoint "$DRAFT_CHECKPOINT" \
    --profile-dir "$OPD_PROPOSAL_PROFILE_DIR" --profile "$OPD_PROPOSAL_PROFILE" \
    --rank "$OPD_RANK" --dtype "$MODEL_DTYPE" --topk "$OPD_TOPK" \
    --auto-tune "$OPD_AUTO_TUNE_IF_MISSING" --batch-size "$BATCH_SIZE" \
    --responses "$RESPONSES_PER_PROMPT" --max-draft-k "$MAX_DRAFT_K" \
    --iterations "${OPD_TUNE_ITERATIONS:-30}")"
  export OPD_PROPOSAL_PROFILE
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RUN_DIR/statistics"
mkdir -p "$TRAIN_MODEL_ROOT"
ln -sfn "$RUN_DIR" "$TRAIN_MODEL_ROOT/active_run"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/write_run_metadata.py" --run-dir "$RUN_DIR" --kind train \
  --item "run_name=$RUN_NAME" --item "model=$MODEL" --item "dataset=$DATASET_PATH" \
  --item "draft_checkpoint=$DRAFT_CHECKPOINT" --item "draft_architecture=FastGRPO" \
  --item "target_adapter=$TARGET_ADAPTER" \
  --item "target_lr=$TARGET_LR" --item "draft_lr=$DRAFT_LR" \
  --item "persistent_draft_objective=fastgrpo_smoothl1_2_ce_0.1" \
  --item "batch_size=$BATCH_SIZE" --item "accumulation_steps=$ACCUMULATION_STEPS" \
  --item "responses_per_prompt=$RESPONSES_PER_PROMPT" --item "temperature=$TEMPERATURE" \
  --item "top_p=$TOP_P" --item "max_length=$GEN_MAX_LENGTH" \
  --item "max_prompt_length=$MAX_PROMPT_LENGTH" --item "num_epochs=$NUM_EPOCHS" \
  --item "resume_checkpoint=$RESUME_CHECKPOINT" --item "method=$METHOD" \
  --item "opd_rank=$OPD_RANK" --item "opd_topk=$OPD_TOPK" \
  --item "opd_fast_lr=$OPD_FAST_LR" --item "opd_update_stream=$OPD_UPDATE_STREAM" \
  --item "opd_projector_lr=$OPD_PROJECTOR_LR" --item "rollout_log_flush_interval=$ROLLOUT_LOG_FLUSH_INTERVAL" \
  --item "opd_visited_weight=$OPD_VISITED_WEIGHT" --item "opd_frontier_weight=$OPD_FRONTIER_WEIGHT" \
  --item "opd_profile=$OPD_PROFILE" --item "opd_diagnostics=$OPD_DIAGNOSTICS" \
  --item "opd_backend=$OPD_BACKEND" \
  --item "opd_ablation_name=$OPD_ABLATION_NAME" --item "opd_projector_init=$OPD_PROJECTOR_INIT" \
  --item "opd_projector_seed=$OPD_PROJECTOR_SEED" --item "opd_update_interval_rounds=$OPD_UPDATE_INTERVAL_ROUNDS" \
  --item "opd_projector_mode=${OPD_PROJECTOR_MODE:-off}" \
  --item "opd_train_projector=$OPD_TRAIN_PROJECTOR" --item "opd_proposal_mode=$OPD_PROPOSAL_MODE" \
  --item "opd_dense_implementation=$OPD_DENSE_IMPLEMENTATION" \
  --item "opd_proposal_profile=$OPD_PROPOSAL_PROFILE" \
  --item "opd_proposal_profile_dir=$OPD_PROPOSAL_PROFILE_DIR" \
  --item "opd_kv_max_retained_tokens=$OPD_KV_MAX_RETAINED_TOKENS" \
  --item "kv_gather_strategy=${KV_GATHER_STRATEGY:-stacked}" \
  --item "nproc_per_node=$NPROC_PER_NODE"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/console.log"
ln -sfn "$RUN_DIR" "$TRAIN_MODEL_ROOT/latest_run"
