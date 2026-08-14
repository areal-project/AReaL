#!/bin/bash
# Submit ALFWorld Qwen3.6 MemRL-trajectory + Mem0 (4x H200).
set -euo pipefail

export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}"
: "${AISTUDIO_TOKEN:?Please export AISTUDIO_TOKEN before submitting}"
export WORKER_NUM=0
export JOB_TAG="${JOB_TAG:-}"
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-qwen36-traj-mem0-${TS}"
export KM_IMAGE="${KM_IMAGE:-acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429}"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_memrl_traj_mem0.sh ${TS}"
export GPU_TYPE=h200
export GPU_NUM=4

uv pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
uv pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_mem0.py
