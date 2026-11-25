#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sweep="$1"
shift
one_gpu
dataset_defaults

MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
one_gpu
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-${PRETRAIN_MODEL_ROOT:-${PRETRAIN_OUTPUT_ROOT:-$PROJECT_DIR/outputs}/pretrain/$MODEL_KEY}/latest_checkpoint}"
export OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$PROJECT_DIR/outputs/benchmarks/opd_proposals}"
export OPD_SAMPLER_MODE="${OPD_SAMPLER_MODE:-finite}"
scope="${OPD_B_SHARING_SCOPE:-per_prompt_group}"
seed="${BENCHMARK_SEED:-42}"
case "$sweep" in
  concurrency)
    if [[ "${PROMPT_GROUP_SWEEP:-false}" == true ]]; then
      sweep=prompt_groups; shape=b1-2-4-8-16_g8_cap"total$GEN_MAX_LENGTH"
    else
      shape=b8_g4-8-16_cap"total$GEN_MAX_LENGTH"
    fi ;;
  long_context) shape="b${BENCHMARK_BATCH_SIZE:-8}_g${BENCHMARK_RESPONSES:-8}_cap2048-4096" ;;
  *) fail "invalid sweep: $sweep" ;;
esac
run="$(new_run_dir "$REBUTTAL_OUTPUT_ROOT/frozen_benchmarks" "${sweep}__${MODEL_KEY}__${DATASET}__fastgrpo-vs-opd_reflex__seed${seed}__${shape}")"
profile="${GENERATION_PROFILE:-detailed}"
cmd=("$PYTHON_BIN" "$PROJECT_DIR/scripts/benchmark_opd_reflex.py" rebuttal --sweep "$sweep"
  --target-model "$MODEL" --model-key "$MODEL_KEY" --draft-checkpoint "$DRAFT_CHECKPOINT"
  --target-adapter "$TARGET_ADAPTER" --projector-checkpoint "${PROJECTOR_CHECKPOINT:-}"
  --dataset "$DATASET" --dataset-path "$DATASET_PATH" --output "$run" --seed "$seed"
  --prompt-seed "${BENCHMARK_PROMPT_SEED:-42}" --batch-size "${BENCHMARK_BATCH_SIZE:-8}"
  --responses "${BENCHMARK_RESPONSES:-8}" --repetitions "${BENCHMARK_REPETITIONS:-5}"
  --warmup "${BENCHMARK_WARMUP:-1}" --generation-profile "$profile" --round-log-frequency "${ROUND_LOG_FREQUENCY:-20}"
  --max-length "$GEN_MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH" --dtype "$MODEL_DTYPE"
  --attn-implementation "$ATTENTION_IMPLEMENTATION" --temperature "$TEMPERATURE" --top-p "$TOP_P"
  --top-k "${TOP_K:-0}" --verification-capacity "$VERIFICATION_CAPACITY" --max-verification-num "$MAX_VERIFICATION_NUM"
  --max-draft-k "$MAX_DRAFT_K" --max-draft-token-length "$MAX_DRAFT_TOKEN_LENGTH"
  --min-draft-token-length "$MIN_DRAFT_TOKEN_LENGTH" --draft-token-length-c "$DRAFT_TOKEN_LENGTH_C"
  --rank "$OPD_RANK" --topk "$OPD_TOPK" --opd-fast-lr "$OPD_FAST_LR"
  --opd-visited-weight "$OPD_VISITED_WEIGHT" --opd-frontier-weight "$OPD_FRONTIER_WEIGHT"
  --opd-update-stream "$OPD_UPDATE_STREAM" --opd-b-sharing-scope "$scope")
[[ "$profile" =~ ^(clean|detailed)$ && "$scope" =~ ^(per_prompt_group|rollout_batch)$ &&
   "${BENCHMARK_REPETITIONS:-5}" =~ ^[1-9][0-9]*$ && "${BENCHMARK_WARMUP:-1}" =~ ^[1-9][0-9]*$ ]] ||
  fail 'set valid profiling/scope options and positive repetition/warmup counts'
printf 'Run dir : %s\n' "$run"
show_command env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
[[ "${DRY_RUN:-false}" == true ]] && exit 0
[[ -f "$MODEL/config.json" ]] || fail "missing model config: $MODEL/config.json"
[[ -e "$DATASET_PATH" ]] || fail "missing benchmark dataset: $DATASET_PATH"
[[ -f "$DRAFT_CHECKPOINT" || -f "$DRAFT_CHECKPOINT/draft.pth" ]] || fail "set DRAFT_CHECKPOINT to a draft.pth file or its parent: $DRAFT_CHECKPOINT"
[[ -z "${PROJECTOR_CHECKPOINT:-}" || -f "$PROJECTOR_CHECKPOINT" || -f "$PROJECTOR_CHECKPOINT/draft.pth" ]] || fail "missing projector checkpoint: $PROJECTOR_CHECKPOINT"
[[ -z "$TARGET_ADAPTER" || -f "$TARGET_ADAPTER/adapter_config.json" ]] || fail "missing target adapter: $TARGET_ADAPTER"
[[ ! -e "$run" ]] || fail "output already exists: $run"
mkdir -p "$(dirname "$run")"
"${cmd[@]}" 2>&1 | tee "$run.console.log"
mv "$run.console.log" "$run/logs/console.log"
