#!/bin/bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory; MEMRL=$ROOT/MemRL; STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=1 MASTER_GPU_NUM=1 AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-current-nomem-val-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.passk10_local.yaml
export BCB_EPOCHS=1
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_current_nomem_val
export BCB_ENTRYPOINT=scripts/run_bcb_nomem_val_only.py
export BCB_NO_AUTORESUME=1
export BCB_EXTRA_ARGS=""
LOG_BASE="$MEMRL/logs/aistudio_bcb_current_nomem_val_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=1 BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_NO_AUTORESUME=1 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"; bash submit.sh
