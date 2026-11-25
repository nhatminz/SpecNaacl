#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sweep="$1"
shift
one_gpu
dataset_defaults

MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
RUNTIME="$PROJECT_DIR/runtimes/specnaacl"
export METHOD=specnaacl
source "${SPECNAACL_COMMON_ENV:-$RUNTIME/configs/_shared/b200_common.env}"
source "${SPECNAACL_MODEL_ENV:-$RUNTIME/configs/$MODEL_KEY/b200.env}"
one_gpu
pretrained="${PRETRAIN_MODEL_ROOT:-${PRETRAIN_OUTPUT_ROOT:-$PROJECT_DIR/outputs}/pretrain/$MODEL_KEY}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$pretrained/latest_checkpoint}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$pretrained/latest_draft_config.json}"
VOCAB_MAPPING="${VOCAB_MAPPING-$pretrained/latest_vocab_mapping.pt}"
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
run="$(new_run_dir "$REBUTTAL_OUTPUT_ROOT/frozen_benchmarks" "${sweep}__${MODEL_KEY}__${DATASET}__fastgrpo-vs-specnaacl__seed${seed}__${shape}")"
profile="${GENERATION_PROFILE:-detailed}"
cmd=("$PYTHON_BIN" "$RUNTIME/benchmark.py" --sweep "$sweep"
  --target-model "$MODEL" --model-key "$MODEL_KEY" --draft-checkpoint "$DRAFT_CHECKPOINT"
  --target-adapter "$TARGET_ADAPTER" --draft-config "$DRAFT_CONFIG" --vocab-mapping "$VOCAB_MAPPING"
  --dataset "$DATASET" --dataset-path "$DATASET_PATH" --output "$run" --seed "$seed"
  --prompt-seed "${BENCHMARK_PROMPT_SEED:-42}" --batch-size "${BENCHMARK_BATCH_SIZE:-8}"
  --responses "${BENCHMARK_RESPONSES:-8}" --repetitions "${BENCHMARK_REPETITIONS:-5}"
  --warmup "${BENCHMARK_WARMUP:-1}" --generation-profile "$profile" --round-log-frequency "${ROUND_LOG_FREQUENCY:-20}"
  --max-length "$GEN_MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH" --dtype "$MODEL_DTYPE"
  --attn-implementation "$ATTENTION_IMPLEMENTATION" --temperature "$TEMPERATURE" --top-p "$TOP_P"
  --top-k "${TOP_K:-0}" --verification-capacity "$VERIFICATION_CAPACITY" --max-verification-num "$MAX_VERIFICATION_NUM"
  --max-draft-k "$MAX_DRAFT_K" --max-draft-token-length "$MAX_DRAFT_TOKEN_LENGTH"
  --min-draft-token-length "$MIN_DRAFT_TOKEN_LENGTH" --draft-token-length-c "$DRAFT_TOKEN_LENGTH_C"
  --reflex-feature-dim "$REFLEX_FEATURE_DIM" --reflex-lr "$REFLEX_LR"
  --reflex-update-stream "$REFLEX_UPDATE_STREAM" --reflex-seed "$REFLEX_SEED"
  --reflex-weight-decay "$REFLEX_WEIGHT_DECAY" --reflex-feedback-scope "$REFLEX_FEEDBACK_SCOPE"
  --reflex-proposal-strategy "$REFLEX_PROPOSAL_STRATEGY" --reflex-correction-strategy "$REFLEX_CORRECTION_STRATEGY"
  --reflex-feedback-strategy "$REFLEX_FEEDBACK_STRATEGY" --reflex-feature-strategy "$REFLEX_FEATURE_STRATEGY")
[[ "$profile" =~ ^(clean|detailed)$ &&
   "${BENCHMARK_REPETITIONS:-5}" =~ ^[1-9][0-9]*$ && "${BENCHMARK_WARMUP:-1}" =~ ^[1-9][0-9]*$ ]] ||
  fail 'set a valid profiling mode and positive repetition/warmup counts'
printf 'Run dir : %s\n' "$run"
show_command env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
[[ "${DRY_RUN:-false}" == true ]] && exit 0
[[ -f "$MODEL/config.json" ]] || fail "missing model config: $MODEL/config.json"
[[ -e "$DATASET_PATH" ]] || fail "missing benchmark dataset: $DATASET_PATH"
[[ -e "$DRAFT_CHECKPOINT" ]] || fail "set DRAFT_CHECKPOINT to a frozen SpecForge EAGLE-3 checkpoint: $DRAFT_CHECKPOINT"
[[ -f "$DRAFT_CONFIG" ]] || fail "missing EAGLE-3 config: $DRAFT_CONFIG"
[[ -z "$VOCAB_MAPPING" || -f "$VOCAB_MAPPING" ]] || fail "missing EAGLE-3 vocabulary mapping: $VOCAB_MAPPING"
export PYTHONPATH="$RUNTIME:$PROJECT_DIR/third_party/SpecForge:$PYTHONPATH"
[[ -z "$TARGET_ADAPTER" || -f "$TARGET_ADAPTER/adapter_config.json" ]] || fail "missing target adapter: $TARGET_ADAPTER"
[[ ! -e "$run" ]] || fail "output already exists: $run"
mkdir -p "$(dirname "$run")"
"${cmd[@]}" 2>&1 | tee "$run.console.log"
mv "$run.console.log" "$run/logs/console.log"
