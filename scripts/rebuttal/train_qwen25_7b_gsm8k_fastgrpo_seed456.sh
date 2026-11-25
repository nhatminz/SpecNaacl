#!/usr/bin/env bash
set -euo pipefail
export SEED=456 DATASET="${DATASET:-gsm8k}" REBUTTAL_EXPERIMENT=multiseed
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../train_qwen25_7b_fastgrpo.sh"
