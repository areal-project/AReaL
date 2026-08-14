#!/bin/bash
# Retry the Qwen72B trajectory/Self-RAG holdout arms from their completed Section-4 checkpoints.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}"
export WORKER_NUM=0
export JOB_TAG="qwen72b-holdout-traj-memrl-selfrag-s5-retry2"
TS="20260729_095007"
RETRY_TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-q72b-holdout-traj-memrl-selfrag-s5-retry2-${RETRY_TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16 EMBED_PORT=19090 MEMRL_PORT=19290 SELFRAG_PORT=19390 NCCL_PORT_MEMRL=29690 NCCL_PORT_SELFRAG=29790 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_holdout_traj_memrl_selfrag.sh ${TS}"
export GPU_NUM=5
export GPU_TYPE=h200
export LAUNCH_CONTAINER_MODE=dev_local
if [[ -n "${AISTUDIO_TOKEN:-}" ]]; then
  aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
else
  echo '[INFO] AISTUDIO_TOKEN is unset; using existing aistudio_user login session.'
fi
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_traj_memrl_selfrag.py
