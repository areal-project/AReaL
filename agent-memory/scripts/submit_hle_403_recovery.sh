#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: $0 region|rag|selfrag|mem0 RUN_ID CHECKPOINT_PATH}"
RID="${2:?usage: $0 region|rag|selfrag|mem0 RUN_ID CHECKPOINT_PATH}"
CKPT="${3:?usage: $0 region|rag|selfrag|mem0 RUN_ID CHECKPOINT_PATH}"
case "$MODE" in region|rag|selfrag|mem0) ;; *) exit 2;; esac
[[ -f "$CKPT/snapshot_meta.json" && -d "$CKPT/local_cache" ]] || { echo 'incomplete checkpoint' >&2; exit 2; }
if [[ "$MODE" == mem0 ]]; then [[ -d "$CKPT/mem0_qdrant" && -f "$CKPT/mem0_id_metadata.json" ]] || { echo 'incomplete Mem0 checkpoint' >&2; exit 2; }; fi
export AISTUDIO_LOGIN_NAME=aistudio AISTUDIO_USERNUMBER=477578 WORKER_NUM=0 JOB_TAG='' JOB_NAME="$RID"
export KM_IMAGE='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'
export JOB_COMMAND="HLE_MODE=$(printf '%q' "$MODE") HLE_RID=$(printf '%q' "$RID") HLE_CKPT=$(printf '%q' "$CKPT") bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_hle_403_recovery_remote.sh"
export PYPAI_HOME=/tmp/pypai_hle_403_${RID}
export TMPDIR=/tmp
export PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-}
export LAUNCH_CONTAINER_MODE=dev_local
python3 /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_hle_template.py
