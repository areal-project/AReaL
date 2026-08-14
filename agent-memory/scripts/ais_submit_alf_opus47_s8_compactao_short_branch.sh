#!/bin/bash
# ============================================================================
# AIStudio 提交：ALFWorld opus-4-7 Region+FS (resume from s5_b110)
# ============================================================================

# --- 登录凭证 ---
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

# --- 作业信息 ---
export WORKER_NUM=0
export JOB_NAME="yl-alf-opus47-s8-compactao-short-$(date +%m%d-%H%M)"
export JOB_TAG="yl-memrl"

# --- 镜像 ---
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# --- 启动命令 ---
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_run_alf_opus47_s8_compactao_short_branch.sh"

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
