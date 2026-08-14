#!/bin/bash
# Fresh Leaf rerun after the source-conserving split-evidence fix.
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=1
export MASTER_GPU_NUM=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-leaf-splitpriorfix-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local_region.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_leaf_splitpriorfix_control
export BCB_NO_AUTORESUME=1
export BCB_LEAF_ONLY=1
export BCB_COMBINED_STATE_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/leaf_splitprior_fix_20260722_state
export BCB_COMBINED_SCRIPT="$MEMRL/scripts/run_bcb_completion_combined.sh"
LOG_BASE="$MEMRL/logs/aistudio_bcb_leaf_splitpriorfix_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_NO_AUTORESUME=1 BCB_LEAF_ONLY=1 BCB_COMBINED_STATE_DIR=\"${BCB_COMBINED_STATE_DIR}\" BCB_COMBINED_SCRIPT=\"${BCB_COMBINED_SCRIPT}\" bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"
bash submit.sh
