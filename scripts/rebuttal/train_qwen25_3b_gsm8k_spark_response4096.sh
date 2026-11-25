#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_qwen25_3b_gsm8k_specnaacl_response4096.sh"
