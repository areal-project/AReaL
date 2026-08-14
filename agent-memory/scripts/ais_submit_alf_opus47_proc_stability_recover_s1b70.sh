#!/bin/bash
# AIS submit: resume the fixed Opus experiment from its own latest complete checkpoint.
set -euo pipefail
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
export WORKER_NUM=0
export JOB_NAME="yl-alf-opus47-proc-stability-recover-s1b70-$(date +%m%d-%H%M)"
export JOB_TAG="yl-memrl"
export AIS_RETRY_POLICY="ON_FAILURE"
export AIS_RETRY_MAX_ATTEMPT="2"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_run_alf_opus47_proc_stability_recover_s1b70.sh"
if command -v uv >/dev/null 2>&1; then
  uv pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
  uv pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
fi
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_submit_alf_opus47_region.py
