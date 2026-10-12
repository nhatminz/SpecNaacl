#!/usr/bin/env bash
# Exactly five variants of the existing learned_random reference pipeline.
set -euo pipefail
ablation="${1:?usage: run_spark_ablation.sh interval5|interval10|per_prompt_group|carry_b|accepted_only [training CLI flags]}"
shift
export OPD_PROJECTOR_INIT=random_orthogonal OPD_PROJECTOR_SEED=42
export OPD_TRAIN_PROJECTOR=1 OPD_UPDATE_INTERVAL_ROUNDS=1
export OPD_B_SHARING_SCOPE=rollout_batch OPD_B_RESET_MODE=rollout
export OPD_VISITED_WEIGHT=1.0 OPD_FRONTIER_WEIGHT=1.0
export OPD_RANK=8 OPD_TOPK=16 OPD_FAST_LR=0.01
export OPD_ABLATION_NAME="$ablation"
case "$ablation" in
  interval5|interval10) export OPD_UPDATE_INTERVAL_ROUNDS="${ablation#interval}" ;;
  per_prompt_group) export OPD_B_SHARING_SCOPE=per_prompt_group ;;
  carry_b) export OPD_B_RESET_MODE=carry ;;
  accepted_only) export OPD_FRONTIER_WEIGHT=0.0 ;;
  *) echo "ERROR: unknown SPARK ablation: $ablation" >&2;exit 2 ;;
esac
export DATASET=simplelr TRAIN_SUBSET_SEED=42
export DRAFT_LR="${DRAFT_LR:-1e-4}"
export OPD_PROJECTOR_LR="${OPD_PROJECTOR_LR:-$DRAFT_LR}"
export MAX_TARGET_OPTIMIZER_STEPS="${MAX_TARGET_OPTIMIZER_STEPS:-1000}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export NPROC_PER_NODE=1 DRAFT_INITIALIZATION_MODE=pretrained IS_TRAIN_DRAFT=True
export REQUIRE_RESUME="${REQUIRE_RESUME:-1}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/train_qwen25_3b.sh" "$@"
