#!/bin/bash
# ============================================================================
# Submit 3 separate resume jobs (Region / MemRL / Pass@10), each 3x H200
# ============================================================================

export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export WORKER_NUM="0"
export JOB_TAG=""
export GPU_NUM=3
export LAUNCH_CONTAINER_MODE=dev_local

if [ -z "${AISTUDIO_USERNUMBER}" ]; then
    echo "错误：AISTUDIO_USERNUMBER 为空" >&2; exit 1
fi
if [ -z "${AISTUDIO_TOKEN}" ]; then
    echo "错误：AISTUDIO_TOKEN 为空" >&2; exit 1
fi

uv pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ 2>/dev/null
uv pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ 2>/dev/null
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ 2>/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ 2>/dev/null

aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"

TS=$(date +%Y%m%d_%H%M%S)

# --- Job 1: Region resume ---
echo "=== Submitting Region resume ==="
export JOB_NAME="yl-alf-qwen72b-region-resume-${TS}"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_region_resume.sh ${TS}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py

# --- Job 2: MemRL resume ---
echo "=== Submitting MemRL resume ==="
export JOB_NAME="yl-alf-qwen72b-memrl-resume-${TS}"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_memrl_resume.sh ${TS}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py

# --- Job 3: Pass@10 resume ---
echo "=== Submitting Pass@10 resume ==="
export JOB_NAME="yl-alf-qwen72b-passk-resume-${TS}"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_passk_resume.sh ${TS}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py

echo "=== All 3 jobs submitted ==="
