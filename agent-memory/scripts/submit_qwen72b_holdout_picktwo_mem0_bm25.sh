#!/bin/bash
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}" AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}" WORKER_NUM=0
TS=$(date +%Y%m%d_%H%M%S)
export JOB_TAG=qwen72b-holdout-picktwo-mem0-bm25
export JOB_NAME="yl-alf-q72b-picktwo-mem0-bm25-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="MEMRL_RUN_ID=qwen72b-picktwo-mem0-bm25-${TS//_/-} MEM0_COLLECTION=memrl_mem0_q72b_picktwo_${TS} bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_holdout_picktwo_mem0_bm25.sh ${TS}"
export GPU_NUM=3 GPU_TYPE=h200 LAUNCH_CONTAINER_MODE=dev_local
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_traj_memrl_selfrag.py
