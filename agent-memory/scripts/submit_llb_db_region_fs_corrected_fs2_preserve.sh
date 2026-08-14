#!/usr/bin/env bash
# Fresh DB main: corrected Region settings, ordinary FS=2, preserve base top-5 selection.
set -euo pipefail
export AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='llb-db-regionfs-corrected-fs2-preserve'
if [[ -z "${AISTUDIO_TOKEN:-}" ]]; then
  token_line=$(grep -m1 '^export AISTUDIO_TOKEN=' /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memp_gpt41mini.sh || true)
  [[ -n "$token_line" ]] || { echo 'ERROR: AISTUDIO_TOKEN unavailable' >&2; exit 1; }
  eval "$token_line"
fi
STAMP=$(date +%Y%m%d-%H%M%S)
export MEMRL_RUN_ID="region-fs-db-gpt41mini-corrected-fs2-preserve-${STAMP}"
export JOB_NAME="yl-llbdb-regionfs-corrected-fs2-preserve-${STAMP}"
export KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="export MEMRL_RUN_ID='${MEMRL_RUN_ID}'; bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_region_fs_corrected_fs2_preserve_aistudio.sh"
export LAUNCH_CONTAINER_MODE=dev_local
pip install 'aistudio-common>=0.0.28.75' -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
printf '%s\n' "[INFO] Submitting DB corrected Region main with ordinary FS2 + selector-ID preservation: $JOB_NAME"
printf '%s\n' "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID (fresh; no backfill; no forced failure slot; base top-5 IDs unchanged)"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_region_fs_stable_fs1_fixed5.py
