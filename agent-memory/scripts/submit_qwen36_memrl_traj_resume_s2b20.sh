#!/bin/bash
# Submit a standalone Qwen3.6 MemRL-trajectory continuation from s2_b20.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"
export WORKER_NUM="0"
export JOB_TAG="qwen36-memrl-traj-resume-s2b20"
export JOB_NAME="yl-alf-qwen36-memrltraj-resume-s2b20-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen36_memrl_traj_resume_s2b20.sh ${TS}"
export GPU_NUM="2"
export GPU_TYPE="h200"
export LAUNCH_CONTAINER_MODE="dev_local"
# pypai is installed in the submitting user's site packages.
export PYTHONPATH="/home/admin/.local/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
# Do not use sudo: it drops the logged-in client's storage endpoint configuration.
/opt/conda/bin/python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen36_main.py
