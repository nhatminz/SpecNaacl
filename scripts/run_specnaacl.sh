#!/usr/bin/env bash
set -euo pipefail

# Treatment: identical runtime/configuration with FastLKReflex enabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD="specnaacl"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
source "$SCRIPT_DIR/launch/train_model.sh" "$@"
