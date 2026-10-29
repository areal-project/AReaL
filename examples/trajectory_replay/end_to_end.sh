#!/usr/bin/env bash
# Real GPU training only; not executed locally. See README.md for acceptance checks.
set -euo pipefail
exec python3 "$(dirname "$0")/run_gpu.py" "$@" --steps 3
