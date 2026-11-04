#!/usr/bin/env bash
# Run a real configured training entrypoint and verify that rollout cost shares
# appear in its logged stats. Pass the complete training command as arguments.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 python3 <training-entrypoint> --config <config.yaml> [overrides...]" >&2
  exit 2
fi

log_file="${ROLLOUT_TIMEPERF_LOG:-$(mktemp)}"
echo "Training log: ${log_file}"
"$@" 2>&1 | tee "${log_file}"

for metric in \
  timeperf/rollout_group_attempt_seconds \
  timeperf/rollout_group_filter_rejected_share \
  timeperf/rollout_group_below_min_share \
  timeperf/rollout_group_other_discarded_share; do
  if ! grep -Fq "${metric}" "${log_file}"; then
    echo "Missing metric: ${metric}. See ${log_file}" >&2
    exit 1
  fi
done
echo "Rollout group timing metrics found in ${log_file}"
