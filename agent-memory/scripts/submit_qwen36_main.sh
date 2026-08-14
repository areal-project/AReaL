#!/bin/bash
# Submit Qwen3.6 ALFWorld MAIN (5 experiments, no mem0) to AIStudio
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-qwen36-main-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_main.sh ${TS}"
export GPU_TYPE="h200"

if [ -z "${AISTUDIO_USERNUMBER}" ]; then echo "错误：AISTUDIO_USERNUMBER 为空" >&2; exit 1; fi
if [ -z "${AISTUDIO_TOKEN}" ]; then echo "错误：AISTUDIO_TOKEN 为空" >&2; exit 1; fi

uv pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ 2>/dev/null
uv pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ 2>/dev/null

export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_main.py
