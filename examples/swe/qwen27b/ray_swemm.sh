#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run as the driver inside an existing Ray cluster or via `ray job submit`.
set -euo pipefail
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: QWEN_ENV_FILE=<shared-profile> bash ray_swemm.sh --config <subtask/train.yaml> [--prepare-only] [--resume] [--set KEY=VALUE ...]'
  exit 0
fi
if [[ ${1:-} != --config || $# -lt 2 ]]; then
  echo 'Pass --config <subtask/train.yaml> first to select the intended SWE-MM streams and rewards.' >&2
  exit 2
fi
export QWEN_SUBMITTED_TASK=swe
exec bash "$recipe/../rl/qwen38_27b/ray_worker.sh" controller "$@"
