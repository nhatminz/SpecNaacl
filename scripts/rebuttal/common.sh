#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/data}"
export REBUTTAL_OUTPUT_ROOT="${REBUTTAL_OUTPUT_ROOT:-$PROJECT_DIR/outputs/rebuttal}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

fail() { echo "ERROR: $*" >&2; exit 2; }

one_gpu() {
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "$NPROC_PER_NODE" == 1 && "$CUDA_VISIBLE_DEVICES" =~ ^[a-zA-Z0-9_/-]+$ ]] ||
    fail "set NPROC_PER_NODE=1 and CUDA_VISIBLE_DEVICES to one GPU index or UUID"
}

show_command() {
  printf 'Command :'; printf ' %q' "$@"; printf '\n'
}

dataset_defaults() {
  export DATASET="${DATASET:-gsm8k}"
  case "${DATASET,,}" in
    gsm8k) export DATASET_PATH="${DATASET_PATH:-${GSM8K_TRAIN:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}}" ;;
    simplelr|simplelr_abel|simplelr_abel_level3to5) export DATASET=simplelr DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
    dapo|dapo-math|dapo_math) export DATASET=dapo DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
    *) [[ -n "${DATASET_PATH:-}" ]] || fail "set DATASET_PATH for DATASET=$DATASET" ;;
  esac
  [[ "$DATASET" =~ ^[a-zA-Z0-9_-]+$ ]] || fail "invalid dataset identifier: $DATASET"
}

new_run_dir() {
  local parent="$1" name="$2"
  printf '%s/%s__%s__%s' "$parent" "$name" "$(date -u +%Y%m%dT%H%M%S)" \
    "$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex[:12])')"
}
