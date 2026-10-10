#!/usr/bin/env bash
# Uses the existing Qwen2.5-3B / SimpleLR training pipeline and all its logs.
set -euo pipefail
ablation="${1:?usage: run_spark_ablation.sh learned_random|frozen_random|interval5|interval10|interval15|interval0 [training CLI flags]}"
shift
export OPD_PROJECTOR_INIT=random_orthogonal
export OPD_PROJECTOR_SEED="${OPD_PROJECTOR_SEED:-42}"
export OPD_TRAIN_PROJECTOR=1
export OPD_UPDATE_INTERVAL_ROUNDS=1
case "$ablation" in
  learned_random) export OPD_ABLATION_NAME=learned_random ;;
  frozen_random) export OPD_ABLATION_NAME=frozen_random;export OPD_TRAIN_PROJECTOR=0 ;;
  interval5|interval10|interval15|interval0)
    export OPD_UPDATE_INTERVAL_ROUNDS="${ablation#interval}"
    export OPD_ABLATION_NAME="learned_random_$ablation" ;;
  *) echo "ERROR: unknown SPARK ablation: $ablation" >&2;exit 2 ;;
esac
export DATASET=simplelr
export OPD_RANK=8
export TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
export DRAFT_LR="${DRAFT_LR:-1e-4}"
export OPD_PROJECTOR_LR="${OPD_PROJECTOR_LR:-$DRAFT_LR}"
export MAX_TARGET_OPTIMIZER_STEPS="${MAX_TARGET_OPTIMIZER_STEPS:-1000}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export NPROC_PER_NODE=1
export DRAFT_INITIALIZATION_MODE=pretrained
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/train_qwen25_3b.sh" "$@"
