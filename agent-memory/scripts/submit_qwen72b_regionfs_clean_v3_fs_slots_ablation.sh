#!/bin/bash
# Submit Qwen2.5-72B ALFWorld Region+FS clean v3 final-checkpoint FS-slot ablation to AIS.
set -euo pipefail

export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
export WORKER_NUM="0"
export JOB_TAG="qwen72b-regionfs-clean-v3-fs-slots-ablation"
TS=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="yl-alf-q72b-regionfs-v3-fs-slots-abl-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
: "${SNAPSHOT_PATH:?Set completed clean-v3 resume snapshot/10}"
export JOB_COMMAND="SNAPSHOT_PATH=${SNAPSHOT_PATH} EMBED_PORT=19080 LLM_PORT=19280 NCCL_PORT=29680 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_regionfs_clean_v3_fs_slots_ablation.sh ${TS}"
export GPU_NUM=3
export GPU_TYPE="h200"

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
export LAUNCH_CONTAINER_MODE=dev_local

aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py
