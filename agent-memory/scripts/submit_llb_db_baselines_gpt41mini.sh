#!/usr/bin/env bash
set -euo pipefail

BASELINE="${1:-${BASELINE:-rag}}"
case "$BASELINE" in rag|selfrag|mem0) ;; *) echo "Usage: $0 <rag|selfrag|mem0>" >&2; exit 2;; esac

export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
# Keep credentials in the existing submission environment/script; never print them.
if [[ -z "${AISTUDIO_TOKEN:-}" ]]; then
  token_line=$(grep -m1 '^export AISTUDIO_TOKEN=' /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memp_gpt41mini.sh || true)
  [[ -n "$token_line" ]] || { echo "ERROR: AISTUDIO_TOKEN unavailable" >&2; exit 1; }
  eval "$token_line"
fi

export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%m%d-%H%M)
RESUME_SUFFIX=""
if [[ "$BASELINE" == "rag" && -n "${RAG_RESUME_CHECKPOINT:-}" ]]; then
  RAG_RESUME_SECTION="${RAG_RESUME_SECTION:-4}"
  RESUME_SUFFIX="-s${RAG_RESUME_SECTION}resume"
  export MEMRL_RUN_ID="${MEMRL_RUN_ID:-rag-db-gpt41mini-s${RAG_RESUME_SECTION}resume-${TS}}"
fi
if [[ "$BASELINE" == "mem0" ]]; then
  # Stable across platform retries; destination snapshots are the resume authority.
  export MEMRL_RUN_ID="${MEMRL_RUN_ID:-mem0-db-gpt41mini-20260719}"
fi
export JOB_NAME="yl-llbdb-${BASELINE}-gpt41mini${RESUME_SUFFIX}-${TS}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
export JOB_COMMAND="BASELINE=${BASELINE} RAG_RESUME_CHECKPOINT=${RAG_RESUME_CHECKPOINT:-} RAG_RESUME_SECTION=${RAG_RESUME_SECTION:-4} MEMRL_RUN_ID=${MEMRL_RUN_ID:-} bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_baselines_gpt41mini_aistudio.sh"
export LAUNCH_CONTAINER_MODE=dev_local

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null 2>&1

echo "[INFO] Submitting baseline=${BASELINE} job=${JOB_NAME}"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_llb_db_memrl_haiku.py
