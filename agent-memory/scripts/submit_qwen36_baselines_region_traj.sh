#!/bin/bash
# Resume Region+FS trajectory + MemRL/MemP/RAG together.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-qwen36-base-regiontraj-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_baselines_region_traj.sh ${TS}"
export GPU_NUM="5"
export GPU_TYPE="h200"

export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
sudo --preserve-env=AISTUDIO_LOGIN_NAME,AISTUDIO_USERNUMBER,AISTUDIO_TOKEN,WORKER_NUM,JOB_TAG,JOB_NAME,KM_IMAGE,JOB_COMMAND,GPU_NUM,GPU_TYPE,LAUNCH_CONTAINER_MODE \
    python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_main.py
