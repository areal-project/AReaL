#!/bin/bash
# Submit both requested AIS jobs. Credentials are read from the existing private submit script.
set -euo pipefail
BASE=/storage/openpsi/users/yl/agent-memory/MemRL/scripts
LOGIN_NAME=$(sed -n 's/^export AISTUDIO_LOGIN_NAME="\(.*\)"/\1/p' "$BASE/ais_submit_alf_opus47_region_traj_resume.sh")
USERNUMBER=$(sed -n 's/^export AISTUDIO_USERNUMBER="\(.*\)"/\1/p' "$BASE/ais_submit_alf_opus47_region_traj_resume.sh")
TOKEN=$(sed -n 's/^export AISTUDIO_TOKEN="\(.*\)"/\1/p' "$BASE/ais_submit_alf_opus47_region_traj_resume.sh")
export AISTUDIO_LOGIN_NAME="$LOGIN_NAME" AISTUDIO_USERNUMBER="$USERNUMBER" AISTUDIO_TOKEN="$TOKEN"
export WORKER_NUM=0 JOB_TAG=yl-memrl
export KM_IMAGE=acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401
export LAUNCH_CONTAINER_MODE=dev_local
export PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-}
aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN" >/dev/null

export JOB_NAME="yl-opus47-s8-short-$(date +%m%d-%H%M)"
export JOB_COMMAND="bash $BASE/ais_run_alf_opus47_s8_short_branch.sh"
(cd /tmp && python "$BASE/ais_submit_alf_opus47_region.py")

export JOB_NAME="yl-opus47-corrected-id-$(date +%m%d-%H%M)"
export JOB_COMMAND="bash $BASE/ais_run_alf_opus47_corrected_id_s6_s8_s10.sh"
(cd /tmp && python "$BASE/ais_submit_alf_opus47_region.py")
