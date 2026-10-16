#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start one worker per allocated node; this script never requests resources.
set -euo pipefail
mode=${1:-train}
if (( $# )); then shift; fi
case "$mode" in
  train) entry=instrumented_sft.py ;;
  full-parity) entry=full_no_mtp_parity.py ;;
  source-export) entry=source_roundtrip_export.py ;;
  recovered-export) entry=export_recovered.py ;;
  *) echo "usage: $0 {train|source-export|recovered-export} [config overrides]" >&2; exit 2 ;;
esac
: "${MODEL_PATH:?Set the source HF checkpoint}"
: "${SFT_DATASET:?Set the tokenized dataset path}"
: "${FILERoot:?Set a dedicated output directory}"
: "${TRIAL_NAME:?Set a unique trial name}"
: "${MASTER_ADDR:?Set the rendezvous address}"
: "${MASTER_PORT:?Set the rendezvous port}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-chucai-rsi-task2-native-mtp}
export TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-5}
export SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-16384}
export RECOVER_MODE=${RECOVER_MODE:-disabled}
export AREAL_SPMD_MODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export USE_MCORE_GDN=1 QSA_SPARSE_KERNEL=0 PLE_FUSED_KERNEL=0 PLE_CPU_OFFLOAD=0
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd -- "$script_dir/../../.." && pwd)
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m torch.distributed.run \
  --nnodes="${NNODES:-2}" --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --node-rank="${NODE_RANK:-0}" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$script_dir/$entry" --config "$script_dir/two_node_sft.yaml" "$@"
