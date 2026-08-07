#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B colocated AdamW DTE validation launcher.
#
# This profile records the validated 2026-08-06 20-step run:
#   Slurm job: 948425
#   trial: qwen3_8b_colocation_adamw_20step_bs4_ns4_0806_220629
#   model: /storage/openpsi/models/Qwen__Qwen3-8B
#   actor.backend=megatron:d1p1t4
#   rollout.backend=sglang:d1t4p1
#   train_dataset.batch_size=4
#   gconfig.n_samples=4
#
# The underlying launcher still supports SUBMIT=0, NODELIST, RESERVATION,
# TRIAL_NAME, FILEROOT, and other explicit env overrides.

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)

usage() {
  cat <<'USAGE'
Usage:
  examples/dte/run_qwen3_8b_colocation_dte_adamw_20step.sh

Useful environment overrides:
  SUBMIT=0                         Generate sbatch without submitting.
  NODELIST=slurmd-24               Pin the Slurm job to a node.
  TRIAL_NAME=qwen3_8b_manual       Set the trial name.
  FILEROOT=/storage/.../fileroot   Set output root.

This profile is intentionally colocated-only and defaults to the validated
Qwen3-8B AdamW DTE 20-step configuration from Slurm job 948425.
USAGE
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

export MODEL_PATH="${MODEL_PATH:-/storage/openpsi/models/Qwen__Qwen3-8B}"
export TOPOLOGY="${TOPOLOGY:-colocation}"
if [[ "${TOPOLOGY}" != "colocation" ]]; then
  echo "This profile is colocated only; got TOPOLOGY=${TOPOLOGY}" >&2
  exit 2
fi

export ACTOR_BACKEND="${ACTOR_BACKEND:-megatron:d1p1t4}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d1t4p1}"
export CLUSTER_GPUS="${CLUSTER_GPUS:-4}"
export SBATCH_GPUS="${SBATCH_GPUS:-4}"
export JOB_NAME="${JOB_NAME:-pyq-dte-q3-8b-col20}"
export EXP_NAME="${EXP_NAME:-pyq-areal-port-dte-qwen3-8b}"
export TRIAL_SUFFIX="${TRIAL_SUFFIX:-$(date +%m%d_%H%M%S)}"
export TRIAL_NAME="${TRIAL_NAME:-qwen3_8b_colocation_adamw_20step_bs4_ns4_${TRIAL_SUFFIX}}"
export FILEROOT="${FILEROOT:-/storage/openpsi/users/pengzai.pyq/areal_port_dte_qwen3_8b/fileroot}"

export TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-20}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export N_SAMPLES="${N_SAMPLES:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export MAX_CONCURRENT_ROLLOUTS="${MAX_CONCURRENT_ROLLOUTS:-16}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.35}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
export DTE_DELTA_METHOD="${DTE_DELTA_METHOD:-adamw}"
export DTE_VERIFY_SNAPSHOT="${DTE_VERIFY_SNAPSHOT:-false}"
export SAVE_FREQ_STEPS="${SAVE_FREQ_STEPS:-null}"

exec "${SCRIPT_DIR}/run_qwen3_0_6b_dte_smoke.sh"
