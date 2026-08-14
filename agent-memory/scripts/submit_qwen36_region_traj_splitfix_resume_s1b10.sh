#!/bin/bash
# Submit Region split-fix continuation from v8 s1_b10 with 5-batch snapshots.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"
export WORKER_NUM="0"
export JOB_TAG="qwen36-region-splitfix-resume-s1b10"
export JOB_NAME="yl-alf-qwen36-regiontraj-splitfix-resume-s1b10-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_region_traj_splitfix_resume_s1b10.sh ${TS}"
export GPU_NUM="2"
export GPU_TYPE="h200"
export LAUNCH_CONTAINER_MODE="dev_local"
export PYTHONPATH="/home/admin/.local/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
/opt/conda/bin/python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_main.py
