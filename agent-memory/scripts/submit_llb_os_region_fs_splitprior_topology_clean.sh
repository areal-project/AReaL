#!/usr/bin/env bash
# Safe submit wrapper: credentials must be supplied by the caller/environment, never stored here.
set -euo pipefail
: "${AISTUDIO_LOGIN_NAME:?export AISTUDIO_LOGIN_NAME before submitting}"
: "${AISTUDIO_USERNUMBER:?export AISTUDIO_USERNUMBER before submitting}"
: "${AISTUDIO_TOKEN:?export AISTUDIO_TOKEN before submitting}"
export LLB_OS_TASK_KEY='llb-os/regionfs/gpt41mini/splitprior-topology-clean'
TS=$(date +%m%d-%H%M)
export JOB_NAME="yl-llbos-regionfs-clean-${TS}"
export JOB_TAG='llb-os-regionfs-splitprior-topology-clean'
export KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export LLB_OS_RUN_ID="${LLB_OS_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export JOB_COMMAND="export MEMRL_RUN_ID='${LLB_OS_RUN_ID}'; bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_os_region_fs_splitprior_topology_clean_aistudio.sh"
pip install 'aistudio-common>=0.0.28.75' -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:/storage/openpsi/users/yl/agent-memory/MemRL/scripts:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_os_mem0_gpt41mini.py
