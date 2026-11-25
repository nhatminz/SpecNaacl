#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" concurrency qwen25_3b specnaacl 42 16 ""
