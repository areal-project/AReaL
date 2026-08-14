#!/bin/bash
# Submit the RESUME of the stalled haiku OS MemRL run to AIStudio.
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%m%d-%H%M)
export JOB_NAME="yl-llbos-memrl-gpt41mini-resume-${TS}"

export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_os_memrl_gpt41mini_resume.sh"

if [ -z "${AISTUDIO_USERNUMBER}" ]; then echo "错误：AISTUDIO_USERNUMBER 为空" >&2; exit 1; fi
if [ -z "${AISTUDIO_TOKEN}" ]; then echo "错误：AISTUDIO_TOKEN 为空" >&2; exit 1; fi

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/

export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"

python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memrl_haiku.py
