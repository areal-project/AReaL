#!/bin/bash
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}" AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}" WORKER_NUM=0
ORIGINAL_TAG=20260729_095007
NOW=$(date +%Y%m%d_%H%M%S)
export JOB_TAG=qwen72b-holdout-pickplace-to-s10
export JOB_NAME="yl-alf-q72b-holdout-pickplace-to-s10-${NOW}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_holdout_traj_memrl_selfrag_to_s10.sh ${ORIGINAL_TAG}"
export GPU_NUM=5 GPU_TYPE=h200 LAUNCH_CONTAINER_MODE=dev_local
if [[ -n "${AISTUDIO_TOKEN:-}" ]]; then aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"; fi
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_traj_memrl_selfrag.py
