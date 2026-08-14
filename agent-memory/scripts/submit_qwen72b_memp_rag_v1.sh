#!/bin/bash
# Submit Qwen2.5-72B ALFWorld MemP + RAG as one 5xH200 AIS job.
set -euo pipefail

export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}"
export WORKER_NUM=0
export JOB_TAG="qwen72b-memp-rag-v1"
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-q72b-memp-rag-v1-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429"
export JOB_COMMAND="MEMP_RUN_ID=qwen72b_memp_v1 RAG_RUN_ID=qwen72b_rag_v1 EMBED_PORT=19110 MEMP_LLM_PORT=19310 RAG_LLM_PORT=19311 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_memp_rag_v1.sh ${TS}"
export GPU_NUM=5
export GPU_TYPE=h200
export LAUNCH_CONTAINER_MODE=dev_local

if [[ -n "${AISTUDIO_TOKEN:-}" ]]; then
    aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
else
    echo "[INFO] AISTUDIO_TOKEN is unset; using the existing aistudio_user login session."
fi

cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py
