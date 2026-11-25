#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" multiseed qwen25_7b opd_reflex 456 8 ""
