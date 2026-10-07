#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPD_PROPOSAL_MODE=sparse  # builder must not consume an existing profile
export OPD_PROPOSAL_PROFILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_KEY="${MODEL_KEY:-qwen3_1p7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
export DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_MODEL_ROOT/latest_draft_config.json}"
[[ -f "$DRAFT_CONFIG" ]] || { echo "Missing production DRAFT_CONFIG: $DRAFT_CONFIG" >&2; exit 1; }
OPD_TUNE_OUTPUT="${OPD_TUNE_OUTPUT:-$OUTPUT_ROOT/benchmarks/opd_proposals/$MODEL_KEY.json}"
exec "$PYTHON_BIN" "$ROOT/scripts/tune_opd_proposals.py" --output "$OPD_TUNE_OUTPUT" \
  --draft-config "$DRAFT_CONFIG" --dtype "${OPD_TUNE_DTYPE:-${MODEL_DTYPE:-bf16}}" \
  --shapes "${OPD_TUNE_SHAPES:-1x1,8x1,32x1,64x1,16x8,32x8,64x8}" --rank "${OPD_RANK:-8}" \
  --iterations "${OPD_TUNE_ITERATIONS:-30}" "$@"
