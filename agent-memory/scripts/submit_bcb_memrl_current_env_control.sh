#!/bin/bash
# Fresh current-environment MemRL control for historical-baseline fairness audit.
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=1
export MASTER_GPU_NUM=1
export AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-memrl-current-control-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_memrl_current_env_control
export BCB_ENTRYPOINT=run/run_bcb.py
export BCB_NO_AUTORESUME=0
# Historical MemRL surface: no Region, no FS, no baseline-mode overrides.
export BCB_EXTRA_ARGS="--n_eval_runs 1"
LOG_BASE="$MEMRL/logs/aistudio_bcb_memrl_current_env_control_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=0 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"
bash submit.sh
