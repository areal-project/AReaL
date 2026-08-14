#!/bin/bash
# Source from an AIS LLB-OS runner after PROJECT_DIR and MEMRL_RUN_ID are set.
# A stable MEMRL_RUN_ID is injected by the submit script. On AIS ON_EVICTION retry,
# the exact same JOB_COMMAND is replayed, so this runner resolves the same ck_dir
# and run_llb.py restores the newest valid completed-section or batch snapshot.
# This helper never fabricates a new run id: that would silently defeat resume.
set -euo pipefail

LLB_OS_EXPERIMENT_NAME="${1:?experiment name required}"
: "${PROJECT_DIR:?PROJECT_DIR must be set}"
: "${MEMRL_RUN_ID:?MEMRL_RUN_ID must be injected by the AIS submit script}"
CKPT_BASE="${MEMRL_CKPT_BASE:-/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect}"
CK_DIR="${CKPT_BASE}/exp_${LLB_OS_EXPERIMENT_NAME}_${MEMRL_RUN_ID}"
SNAP_ROOT="${CK_DIR}/snapshot"

if [[ -d "${SNAP_ROOT}" ]]; then
  LATEST=$(find "${SNAP_ROOT}" -mindepth 1 -maxdepth 1 -type d     \( -name '[0-9]*' -o -name '[0-9]*_b[0-9]*' \)     -exec test -f '{}/snapshot_meta.json' ';' -printf '%f\n' 2>/dev/null     | sort -V | tail -n 1 || true)
  if [[ -n "${LATEST}" ]]; then
    echo "[AUTO-RESUME] run_id=${MEMRL_RUN_ID} experiment=${LLB_OS_EXPERIMENT_NAME} ck_dir=${CK_DIR} latest_snapshot=${LATEST}"
    echo "[AUTO-RESUME] run_llb.py will restore the newest valid snapshot; batch checkpoint names N_bM resume section N at batch M+1."
  else
    echo "[AUTO-RESUME] run_id=${MEMRL_RUN_ID} found checkpoint root but no valid snapshot; clean start."
  fi
else
  echo "[AUTO-RESUME] run_id=${MEMRL_RUN_ID} clean checkpoint root=${CK_DIR}"
fi
