#!/bin/bash
# Submit corrected Qwen2.5-72B Region+FS Cell A as one 3xH200 AIS job.
set -euo pipefail

export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}"
export WORKER_NUM=0
export JOB_TAG="qwen72b-regionfs-cell-a-splitfix"
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-q72b-regionfs-cell-a-splitfix-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="EMBED_PORT=19090 LLM_PORT=19290 NCCL_PORT=29690 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_regionfs_cell_a_splitfix.sh ${TS}"
export GPU_NUM=3
export GPU_TYPE=h200
export LAUNCH_CONTAINER_MODE=dev_local

if [[ -n "${AISTUDIO_TOKEN:-}" ]]; then
    aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
else
    echo "[INFO] AISTUDIO_TOKEN is unset; using the existing aistudio_user login session."
fi

cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py
