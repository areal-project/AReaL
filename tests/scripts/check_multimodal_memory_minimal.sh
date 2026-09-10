#!/usr/bin/env bash
# Run in the target AReaL/Megatron environment; does not allocate GPUs.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
test_tmp="$(mktemp -d)"
trap 'rm -rf "$test_tmp"' EXIT
"${PYTHON:-python}" -m pytest -q --basetemp="$test_tmp" \
  tests/test_broadcast_tensor_container.py \
  tests/infra/rpc/test_engine_validation.py \
  tests/test_megatron_engine_vlm.py \
  tests/test_microbatch_streaming.py
