#!/usr/bin/env bash
# Historical FastLKReflex/EAGLE-3 runtime, isolated from the full-vocabulary trainer.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../rebuttal/common.sh"

RUNTIME="$PROJECT_DIR/runtimes/specnaacl"
export METHOD="${METHOD:-specnaacl}" MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
source "${SPECNAACL_COMMON_ENV:-$RUNTIME/configs/_shared/b200_common.env}"
source "${SPECNAACL_MODEL_ENV:-$RUNTIME/configs/$MODEL_KEY/b200.env}"
one_gpu
dataset_defaults
case "$DATASET" in
  simplelr) train_option=simplelr_abel_level3to5 ;;
  dapo) train_option=DAPO-math ;;
  *) train_option="$DATASET" ;;
esac
case "$METHOD" in
  specnaacl) reflex_mode=active ;;
  fastgrpo) reflex_mode=off ;;
  *) fail "historical runtime supports METHOD=specnaacl or METHOD=fastgrpo" ;;
esac
case "${DRAFT_INITIALIZATION_MODE:-pretrained}" in
  pretrained|random) draft_mode="${DRAFT_INITIALIZATION_MODE:-pretrained}" ;;
  *) fail 'DRAFT_INITIALIZATION_MODE must be pretrained or random' ;;
esac
[[ -z "${RESUME:-}${RUN_DIR:-}${RUN_NAME:-}" ]] ||
  fail 'fresh training only. Unset RESUME, RUN_DIR and RUN_NAME to create a new run'

seed="${SEED:-$TRAIN_SUBSET_SEED}"
order_seed="${PROMPT_ORDER_SEED:-$TRAIN_SUBSET_SEED}"
cap="${MAX_RESPONSE_TOKENS:-}"
[[ -z "$cap" || "$cap" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_RESPONSE_TOKENS must be a positive integer'
eval_path="${EVAL_DATASET_PATH:-${GSM8K_TEST:-$DATA_ROOT/gsm8k/main/test-00000-of-00001.parquet}}"
experiment="${REBUTTAL_EXPERIMENT:-fresh}"
run="$(new_run_dir "$REBUTTAL_OUTPUT_ROOT/$experiment" "${experiment}__${MODEL_KEY}__${DATASET}__engine-eagle3__method-${METHOD}__seed${seed}__b${BATCH_SIZE}__g${RESPONSES_PER_PROMPT}__cap${cap:-total$GEN_MAX_LENGTH}")"
logs="$run/logs"
checkpoints="$run/checkpoints"
pretrained="${PRETRAIN_MODEL_ROOT:-${PRETRAIN_OUTPUT_ROOT:-$PROJECT_DIR/outputs}/pretrain/$MODEL_KEY}"
write_config=false
if [[ "$draft_mode" == pretrained ]]; then
  draft="${DRAFT_CHECKPOINT:-$pretrained/latest_checkpoint}"
  draft_config="${DRAFT_CONFIG:-$pretrained/latest_draft_config.json}"
  vocab="${VOCAB_MAPPING-$pretrained/latest_vocab_mapping.pt}"
else
  draft=
  draft_config="${DRAFT_CONFIG:-$run/draft_config.json}"
  vocab="${VOCAB_MAPPING:-}"
  [[ -n "${DRAFT_CONFIG:-}" ]] || write_config=true
  [[ "$write_config" == false || -z "$vocab" ]] ||
    fail 'set DRAFT_CONFIG together with VOCAB_MAPPING for a random compact draft'
fi
profile="${GENERATION_PROFILE:-clean}"
[[ "$profile" == clean || "$profile" == detailed ]] || fail 'GENERATION_PROFILE must be clean or detailed'
# Use the original, asynchronous Reflex event profiling in diagnostic runs.
if [[ "$profile" == detailed ]]; then REFLEX_PROFILE=1; fi
cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 "$RUNTIME/grpo_speculative.py"
  --method "$METHOD" --reflex_mode "$reflex_mode" --model_dir "$MODEL" --model_type "$MODEL_TYPE"
  --adapter_path "$draft" --draft_config "$draft_config" --vocab_mapping "$vocab"
  --draft_backend eagle3 --draft_initialization_mode "$draft_mode" --load_lora_path "$TARGET_ADAPTER"
  --dataset_path "$DATASET_PATH" --train_option "$train_option" --eval_dataset_path "$eval_path"
  --version_name "$(basename "$run")" --seed "$seed" --train_subset_seed "$TRAIN_SUBSET_SEED"
  --prompt_order_seed "$order_seed" --eval_interval_updates "${EVAL_INTERVAL_UPDATES:-50}"
  --evaluation_seed "${EVALUATION_SEED:-42}" --rebuttal_experiment "$experiment" --rebuttal_model_key "$MODEL_KEY"
  --generation_profile "$profile" --round_log_frequency "${ROUND_LOG_FREQUENCY:-20}"
  --rollout_log_flush_interval "${ROLLOUT_LOG_FLUSH_INTERVAL:-10}"
  --train_data_fraction "$TRAIN_DATA_FRACTION" --max_train_samples "$MAX_TRAIN_SAMPLES"
  --num_epochs "$NUM_EPOCHS" --sample_num "$SAMPLE_NUM" --grpo_iteration_num "$GRPO_ITERATION_NUM"
  --batch_size "$BATCH_SIZE" --repeated_generate_nums "$RESPONSES_PER_PROMPT"
  --accumulation_steps "$ACCUMULATION_STEPS" --draft_accumulation_steps "$DRAFT_ACCUMULATION_STEPS"
  --target_lr "$TARGET_LR" --draft_lr "$DRAFT_LR" --is_train_draft "$IS_TRAIN_DRAFT"
  --draft_train_mode "$DRAFT_TRAIN_MODE" --draft_train_max_batch_size "$DRAFT_TRAIN_MAX_BATCH_SIZE"
  --draft_train_max_tokens "$DRAFT_TRAIN_MAX_TOKENS" --draft_train_max_padding_ratio "$DRAFT_TRAIN_MAX_PADDING_RATIO"
  --draft_train_profile "$DRAFT_TRAIN_PROFILE" --max_target_optimizer_steps "${MAX_TARGET_OPTIMIZER_STEPS:-0}"
  --max_rollout_prompts "${MAX_ROLLOUT_PROMPTS:-0}" --beta "$BETA" --epsilon "$EPSILON"
  --dtype "$MODEL_DTYPE" --attn_implementation "$ATTENTION_IMPLEMENTATION"
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --max_length "$GEN_MAX_LENGTH"
  --max_prompt_length "$MAX_PROMPT_LENGTH" --max_training_token "$MAX_TRAINING_TOKEN"
  --max_training_padding_gap "$MAX_TRAINING_PADDING_GAP" --logps_chunk_size "$LOGPS_CHUNK_SIZE"
  --verification_capacity "$VERIFICATION_CAPACITY" --max_verification_num "$MAX_VERIFICATION_NUM"
  --max_draft_token_length "$MAX_DRAFT_TOKEN_LENGTH" --max_draft_k "$MAX_DRAFT_K"
  --min_draft_token_length "$MIN_DRAFT_TOKEN_LENGTH" --draft_token_length_c "$DRAFT_TOKEN_LENGTH_C"
  --reflex_feature_dim "$REFLEX_FEATURE_DIM" --reflex_lr "$REFLEX_LR" --reflex_weight_decay "$REFLEX_WEIGHT_DECAY"
  --reflex_seed "$REFLEX_SEED" --reflex_feedback_scope "$REFLEX_FEEDBACK_SCOPE" --reflex_backend "$REFLEX_BACKEND"
  --reflex_proposal_strategy "$REFLEX_PROPOSAL_STRATEGY" --reflex_correction_strategy "$REFLEX_CORRECTION_STRATEGY"
  --reflex_feedback_strategy "$REFLEX_FEEDBACK_STRATEGY" --reflex_feature_strategy "$REFLEX_FEATURE_STRATEGY"
  --reflex_profile "$REFLEX_PROFILE" --reflex_diagnostics "$REFLEX_DIAGNOSTICS" --reflex_update_stream "$REFLEX_UPDATE_STREAM"
  --kv_gather_strategy "$KV_GATHER_STRATEGY" --statistical_time "$STATISTICAL_TIME"
  --num_workers "$NUM_WORKERS" --persistent_workers "$PERSISTENT_WORKERS" --log_interval "$LOG_INTERVAL"
  --log_file "$logs/metrics.jsonl" --timing_file "$logs/timing.csv" --summary_file "$run/summary.json"
  --saved_model_dir "$checkpoints/target" --saved_draft_model_dir "$checkpoints/draft"
  --saved_statistics_dir "$run/statistics" --checkpoint_dir "$checkpoints/resume" --resume_checkpoint ''
  --save_checkpoint_steps "$SAVE_CHECKPOINT_STEPS" --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS")
[[ -z "$cap" ]] || cmd+=(--max_response_tokens "$cap")
if [[ "$write_config" == true ]]; then
  printf 'Draft config : generate from target config with identity vocabulary at %s\n' "$draft_config"
fi

printf 'Run dir : %s\nModel : %s\nDataset : %s\nMethod : %s\nEngine : EAGLE-3 / FastLKReflex %s\nDraft init : %s\nDraft checkpoint : %s\nResume : fresh\n' \
  "$run" "$MODEL" "$DATASET_PATH" "$METHOD" "$reflex_mode" "$draft_mode" "${draft:-none}"
show_command env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
[[ "${DRY_RUN:-false}" == true ]] && exit 0

[[ -f "$MODEL/config.json" ]] || fail "missing target config: $MODEL/config.json"
[[ -e "$DATASET_PATH" ]] || fail "missing training dataset: $DATASET_PATH"
[[ -e "$eval_path" ]] || fail "missing held-out GSM8K file: $eval_path"
[[ -z "$TARGET_ADAPTER" || -f "$TARGET_ADAPTER/adapter_config.json" ]] || fail "missing target adapter: $TARGET_ADAPTER"
if [[ "$draft_mode" == pretrained ]]; then
  [[ -e "$draft" ]] ||
    fail "missing EAGLE-3 initialization: $draft. Set DRAFT_CHECKPOINT, DRAFT_CONFIG and VOCAB_MAPPING to the original SpecForge artifacts, or explicitly choose DRAFT_INITIALIZATION_MODE=random for a separate random-init experiment"
  if [[ -f "$draft/draft.pth" ]]; then
    fail "historical SpecNaacl requires a SpecForge EAGLE-3 checkpoint or training_state.pt: $draft"
  fi
fi
[[ "$write_config" == true || -f "$draft_config" ]] || fail "missing EAGLE-3 draft config: $draft_config (set DRAFT_CONFIG)"
[[ -z "$vocab" || -f "$vocab" ]] || fail "missing SpecForge vocab mapping: $vocab (set VOCAB_MAPPING)"
export PYTHONPATH="$RUNTIME:$PROJECT_DIR/third_party/SpecForge:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --python-only
"$PYTHON_BIN" -c 'import sys; sys.path.insert(0, sys.argv[1]); import torch; from helper.eagle3_specforge import require_specforge; require_specforge(); assert torch.cuda.device_count() == 1, "Expose exactly one CUDA GPU"' "$RUNTIME"

mkdir -p "$(dirname "$run")"
mkdir "$run"
mkdir "$logs" "$checkpoints"
if [[ "$write_config" == true ]]; then
  vocab_size="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vocab_size"])' "$MODEL/config.json")"
  "$PYTHON_BIN" "$RUNTIME/scripts/generate_eagle3_config.py" --target-model-path "$MODEL" \
    --draft-vocab-size "$vocab_size" --output "$draft_config"
fi
"${cmd[@]}" 2>&1 | tee "$logs/console.log"
