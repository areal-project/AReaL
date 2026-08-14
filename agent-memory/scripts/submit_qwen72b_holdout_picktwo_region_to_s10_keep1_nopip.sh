#!/bin/bash
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}" AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}" WORKER_NUM=0
ORIGINAL_TAG=20260731_101105; NOW=$(date +%Y%m%d_%H%M%S)
export JOB_TAG=qwen72b-holdout-picktwo-region-e10-nopip
export JOB_NAME="yl-alf-q72b-picktwo-region-e10-nopip-${NOW}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="EMBED_PORT=21190 REGION_PORT=21290 NCCL_REGION=31690 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_holdout_picktwo_region_to_s10_keep1_nopip.sh ${ORIGINAL_TAG}"
export GPU_NUM=3 GPU_TYPE=h200 LAUNCH_CONTAINER_MODE=dev_local
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_traj_memrl_selfrag.py
