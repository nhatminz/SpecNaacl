#!/usr/bin/env bash
set -euo pipefail
export METHOD=specnaacl
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch/train_specnaacl.sh"
