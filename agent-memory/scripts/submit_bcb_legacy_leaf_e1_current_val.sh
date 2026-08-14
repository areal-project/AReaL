#!/bin/bash
# Re-evaluate the historical 54.1 Leaf E1 checkpoint with current serving/evaluator.
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260701_145908_deepseek-ai_DeepSeek-V3_region/epoch1/snapshot/1
[ -f "$SNAP/snapshot_meta.json" ] || { echo "missing historical Leaf E1 snapshot: $SNAP" >&2; exit 1; }
export WORKER_NUM=1 MASTER_GPU_NUM=1 AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-legacy-leaf-e1-current-val-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local_region.yaml
export BCB_EPOCHS=1
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_legacy_leaf_e1_current_val
export BCB_ENTRYPOINT=run/run_bcb_region.py
export BCB_NO_AUTORESUME=1
# Historical Leaf inference surface. No training/topology edit occurs in eval_only.
export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 1 --eval_only --task_cluster_k 0 --region_gating_mode additive --region_utility_mode beta --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 12 --region_min_samples 0 --region_cluster_selection_method leaf --region_max_region_share 0.30 --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 --failure_summary_n_slots 1"
LOG_BASE="$MEMRL/logs/aistudio_bcb_legacy_leaf_e1_current_val_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=1 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"
bash submit.sh
