#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" multiseed qwen25_7b fastgrpo 123 8 ""
