#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
exec singularity exec --nv --no-home --writable-tmpfs --bind "$QWEN_MOUNTS" \
  --env PYTHONPATH="$QWEN_SFT_OVERLAY:$QWEN_SFT_MCORE/src:$QWEN_SFT_ROOT" \
  --env AREAL_SPMD_MODE=1 --env AREAL_DIR="$QWEN_SFT_ROOT" \
  --env MCORE_BRIDGE_ROOT="$QWEN_SFT_MCORE" --env PLE_CPU_OFFLOAD=0 \
  --env TMPDIR=/tmp --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$QWEN_SFT_IMAGE" bash -c '
    cd "$1"
    exec python3 -m torch.distributed.run \
      --nnodes="$2" --nproc-per-node=8 --node-rank="$3" \
      --master-addr="$4" --master-port="$5" \
      examples/swe/train_sft.py --config "$6"
  ' bash "$QWEN_SFT_ROOT" "$N_NODES" "$SLURM_PROCID" "$MASTER_ADDR" "$MASTER_PORT" "$CONFIG"
