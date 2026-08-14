#!/bin/bash
# Submit LLB OS MemRL Region+FS (claude-haiku-4-5) to AIStudio.
# HOLD: only submit when the MemRL run is nearly done.
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%m%d-%H%M)
export JOB_NAME="yl-llbos-regionfs-haiku-${TS}"

export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_os_region_fs_haiku_aistudio.sh"

if [ -z "${AISTUDIO_USERNUMBER}" ]; then echo "错误：AISTUDIO_USERNUMBER 为空" >&2; exit 1; fi
if [ -z "${AISTUDIO_TOKEN}" ]; then echo "错误：AISTUDIO_TOKEN 为空" >&2; exit 1; fi

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/

export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"

python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memrl_haiku.py
