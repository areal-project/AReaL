#!/bin/bash
# ============================================================================
# AIStudio 提交：Opus clean-reset cap128; automatic resume from newest s9_bN if present
# ============================================================================

# --- 登录凭证 ---
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

# --- 作业信息 ---
export WORKER_NUM=0
export JOB_NAME="yl-alf-opus47-s9b30-state-guard-b31b40-$(date +%m%d-%H%M)"
export JOB_TAG="yl-memrl"
export AIS_RETRY_POLICY="NEVER"
export AIS_RETRY_MAX_ATTEMPT="1"

# --- 镜像 ---
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# --- 启动命令 ---
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_run_alf_opus47_s9b30_state_guard_b31_b40.sh"

# --- 提交 ---
if [ -z "${AISTUDIO_USERNUMBER}" ]; then
    echo "错误：AISTUDIO_USERNUMBER 为空" >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
    uv pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
else
    echo "[INFO] uv not found; using preinstalled /tmp/yl_pypai submission environment"
fi

export LAUNCH_CONTAINER_MODE=dev_local

aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"

cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_submit_alf_opus47_region.py
