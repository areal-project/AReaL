#!/usr/bin/env bash
# Requires free GPUs and an existing supported VLM checkpoint/environment.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${VLM_MODEL_PATH:?Set VLM_MODEL_PATH to an existing supported VLM checkpoint}"
parallel_backend="${BACKEND:-megatron:d1p1t2}"
world_size="$("${PYTHON:-python}" -c 'from areal.api.alloc_mode import ModelAllocation; import sys; print(ModelAllocation.from_str(sys.argv[1]).parallel.world_size)' "$parallel_backend")"
result_dir="$(mktemp -d)"
trap 'rm -rf "$result_dir"' EXIT
"${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$world_size" \
  tests/torchrun/run_megatron_engine_vlm_distributed.py \
  --backend="$parallel_backend" --test_type=memory_isolation \
  --output="$result_dir/result.txt"
[[ "$(cat "$result_dir/result.txt")" == Passed ]]
