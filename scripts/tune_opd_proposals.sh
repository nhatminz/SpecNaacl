#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPD_PROPOSAL_MODE=sparse  # builder must not consume an existing profile
export OPD_PROPOSAL_PROFILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
OPD_TUNE_OUTPUT="${OPD_TUNE_OUTPUT:-$ROOT/outputs/benchmarks/opd_proposal_$(date -u +%Y%m%dT%H%M%S_%N).json}"
exec "$PYTHON_BIN" "$ROOT/scripts/tune_opd_proposals.py" --output "$OPD_TUNE_OUTPUT" \
  --shapes "${OPD_TUNE_SHAPES:-64x1,64x7,64x8,32x1,32x8}" --vocab "${OPD_TUNE_VOCAB:-32768}" \
  --hidden "${OPD_TUNE_HIDDEN:-2048}" --rank "${OPD_RANK:-8}" \
  --iterations "${OPD_TUNE_ITERATIONS:-30}" "$@"
