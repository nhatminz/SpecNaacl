#!/usr/bin/env bash
set -euo pipefail

# Fair in-repository baseline: identical SpecNaacl runtime with Reflex disabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD="fastgrpo"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
source "$SCRIPT_DIR/launch/train_model.sh" "$@"
