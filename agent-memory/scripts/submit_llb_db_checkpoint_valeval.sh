#!/usr/bin/env bash
set -euo pipefail
LABEL=${1:?usage: $0 <label> <snapshot> <config> <section>}
SNAPSHOT=${2:?}
CONFIG=${3:?}
SECTION=${4:?}
export AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG=''
if [[ -z "${AISTUDIO_TOKEN:-}" ]]; then
 token_line=$(grep -m1 '^export AISTUDIO_TOKEN=' /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memp_gpt41mini.sh || true)
 [[ -n "$token_line" ]] || { echo 'ERROR: AISTUDIO_TOKEN unavailable' >&2; exit 1; }
 eval "$token_line"
fi
export KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_NAME="yl-llbdb-${LABEL}-readonly-valeval-$(date +%m%d-%H%M)"
export JOB_COMMAND="SOURCE_SNAPSHOT=${SNAPSHOT} SOURCE_CONFIG=${CONFIG} EVAL_SECTION=${SECTION} EVAL_LABEL=${LABEL} bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_checkpoint_valeval_aistudio.sh"
export LAUNCH_CONTAINER_MODE=dev_local
pip install 'aistudio-common>=0.0.28.75' -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memrl_haiku.py
