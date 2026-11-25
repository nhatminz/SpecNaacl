#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_qwen25_7b_gsm8k_specnaacl_seed456.sh"
