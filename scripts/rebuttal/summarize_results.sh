#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
output="${SUMMARY_OUTPUT_DIR:-$(new_run_dir "$REBUTTAL_OUTPUT_ROOT/summaries" multiseed)}"
cmd=("$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_rebuttal.py"
  --runs-root "${TRAIN_RUNS_ROOT:-$REBUTTAL_OUTPUT_ROOT/multiseed}" --output "$output")
[[ -z "${HISTORICAL_MANIFEST:-}" ]] || cmd+=(--historical-manifest "$HISTORICAL_MANIFEST")
cmd+=("$@")
show_command "${cmd[@]}"
[[ "${DRY_RUN:-false}" == true ]] && exit 0
[[ ! -e "$output" ]] || fail "summary output already exists: $output"
exec "${cmd[@]}"
