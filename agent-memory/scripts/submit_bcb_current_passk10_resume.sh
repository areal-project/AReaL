#!/bin/bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory; MEMRL=$ROOT/MemRL; STAMP=$(date +%Y%m%d_%H%M%S)
SOURCE=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_current_passk10/bigcodebench_eval/instruct_full/memory/20260808_060645_deepseek-ai_DeepSeek-V3_rl-off/baseline_passk/results.jsonl
[ -f "$SOURCE" ] || { echo "missing $SOURCE" >&2; exit 1; }
export WORKER_NUM=1 MASTER_GPU_NUM=1 AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-current-passk10-resume-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.passk10_local.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_current_passk10_resume
export BCB_ENTRYPOINT=run/run_bcb.py
export BCB_NO_AUTORESUME=1
export BCB_EXTRA_ARGS="--baseline_mode passk --baseline_k 10 --baseline_resume_results $SOURCE"
LOG_BASE="$MEMRL/logs/aistudio_bcb_current_passk10_resume_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=1 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"; bash submit.sh
