#!/bin/bash
set -euo pipefail
export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-aistudio}" AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-477578}" WORKER_NUM=0
NOW=$(date +%Y%m%d_%H%M%S)
export JOB_TAG=qwen72b-holdout-picktwo-region-quality-fixed-e10
export JOB_NAME="yl-alf-q72b-picktwo-region-quality-fixed-e10-${NOW}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="EMBED_PORT=22390 REGION_PORT=22490 NCCL_REGION=32890 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_holdout_picktwo_region_quality_fixed_e10.sh ${NOW}"
export GPU_NUM=3 GPU_TYPE=h200 LAUNCH_CONTAINER_MODE=dev_local
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_traj_memrl_selfrag.py
