#!/bin/bash
# Submit Qwen3.6 ALFWorld: fresh MemRL-trajectory + MemP continuation (3x H200).
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}"
: "${AISTUDIO_TOKEN:?Please export AISTUDIO_TOKEN before submitting}"
TS=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=0
export JOB_TAG="${JOB_TAG:-qwen36-memrl-traj-memp-resume}"
export JOB_NAME="yl-alf-qwen36-memrltraj-mempresume-${TS}"
export KM_IMAGE="${KM_IMAGE:-acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429}"
export GPU_TYPE=h200
export GPU_NUM=3
export LAUNCH_CONTAINER_MODE=dev_local
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_memrl_traj_memp_resume.sh ${TS}"
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_selfrag.py
