#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=qwen38-sft
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1200G
#SBATCH --time=04:00:00
#SBATCH --output=qwen38-sft-%j.log
set -euo pipefail
# Supply partition/reservation/nodelist/output via sbatch arguments.
: "${AREAL_DIR:?Set repository path}" "${MCORE_BRIDGE_ROOT:?Set bridge path}"
: "${AREAL_IMAGE:?Set actor image}" "${TRAIN_RUNTIME_DEPS:?Set runtime overlay}"
: "${MODEL_PATH:?Set model path}" "${FILERoot:?Set output directory}"
: "${QWEN_MOUNTS:?Set shared filesystem bind mounts}"
export AREAL_DIR MCORE_BRIDGE_ROOT MODEL_PATH FILERoot QWEN_MOUNTS
ROOT=$AREAL_DIR
MCORE=$MCORE_BRIDGE_ROOT
IMAGE=$AREAL_IMAGE
OVERLAY=$TRAIN_RUNTIME_DEPS
mkdir -p "$FILERoot"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head=${nodes[0]}
ip=$(srun --mpi=none --nodes=1 --ntasks=1 --nodelist="$head" hostname --ip-address | awk '{print $1}')
port=${MASTER_PORT:-29541}
export QWEN_SFT_ROOT="$ROOT" QWEN_SFT_MCORE="$MCORE" QWEN_SFT_IMAGE="$IMAGE"
export QWEN_SFT_OVERLAY="$OVERLAY" MASTER_ADDR="$ip" MASTER_PORT="$port"
export N_NODES=${SLURM_JOB_NUM_NODES}
export PP_SIZE=${PP_SIZE:-${SLURM_JOB_NUM_NODES}}
if [[ ${CP_SIZE:-1} != 1 ]]; then
  echo 'This recipe preserves the CP1 baseline; validate CP dependencies separately.' >&2
  exit 2
fi
export MAX_LENGTH=${MAX_LENGTH:-16384} MAX_TOKENS_PER_MB=${MAX_TOKENS_PER_MB:-16384}
CONFIG=${SFT_CONFIG:-examples/swe/qwen38_flash_next/sft_qwen38_flash_next.yaml}
export QWEN_SFT_ROOT="$ROOT" QWEN_SFT_MCORE="$MCORE" QWEN_SFT_IMAGE="$IMAGE"
export QWEN_SFT_OVERLAY="$OVERLAY" CONFIG N_NODES MASTER_ADDR MASTER_PORT
srun --mpi=none --kill-on-bad-exit=1 --ntasks="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 \
  bash "$ROOT/examples/swe/qwen38_flash_next/sft_worker.sh"
