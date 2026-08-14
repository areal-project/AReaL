#!/bin/bash
# Submit corrected LLB OS MemP S7-validation/S8-S10 resume job to AIS.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
: "${AISTUDIO_TOKEN:?AISTUDIO_TOKEN must be set in the environment}"
export WORKER_NUM="0"
export JOB_TAG=""
export LLB_OS_TASK_KEY="llb-os/memp/gpt41mini"
TS=$(date +%m%d-%H%M)
export JOB_NAME="yl-llbos-memp-gpt41mini-resume-s7-${TS}"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_os_memp_gpt41mini_resume_s7_aistudio.sh"
export LAUNCH_CONTAINER_MODE=dev_local

aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memrl_haiku.py
