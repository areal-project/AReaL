#!/bin/bash
# Standalone MemRL E10 standard val + 3-run multi-eval.
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=1
export MASTER_GPU_NUM=1
# Do not let master/worker independently retry into mismatched attempts.
export AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-memrl-e10-multieval-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_memrl_eval_control
export BCB_NO_AUTORESUME=1
export BCB_ENTRYPOINT=scripts/run_bcb_multi_eval_only.py
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_memrl/bigcodebench_eval/instruct_full/memory/20260619_165450_deepseek-ai_DeepSeek-V3_rl-on/epoch10/snapshot/10
[ -f "$SNAP/snapshot_meta.json" ] || { echo "missing MemRL E10 snapshot: $SNAP" >&2; exit 1; }
export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 10 --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
LOG_BASE="$MEMRL/logs/aistudio_bcb_memrl_e10_multieval_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=1 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"
bash submit.sh
