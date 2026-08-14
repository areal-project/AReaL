#!/bin/bash
# Canonical BCB Capability-Region + Region Failure Abstraction (fresh E1).
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)

export WORKER_NUM=1 MASTER_GPU_NUM=1 AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-regionfs-forced-resume304-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local_region.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_region_fs_forced_resume304
export BCB_ENTRYPOINT=run/run_bcb_region.py
export BCB_NO_AUTORESUME=0

# Method invariants follow docs/method_region_fs_canonical.md:
# capability clustering; Beta Region utility; top-3 soft hierarchical shrinkage;
# source-conserving evidence; fixed-K Region failure replacement; coarse-boundary
# topology only; one complete cooldown epoch after each topology edit.
# Remaining numeric values are BCB-adapted scale parameters.
export BCB_EXTRA_ARGS="--resume_from /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_region_fs_canonical_forcedfs5/bigcodebench_eval/instruct_full/region/20260810_090128_deepseek-ai_DeepSeek-V3_region/epoch1/snapshot/step_304 --resume_epoch 1 --resume_step 304 --task_cluster_k 0 --region_gating_mode additive --region_utility_mode beta --region_split_evidence_migration_mode soft_source_conserving --region_temperature 0.1 --shrinkage_top_n 3 --region_min_cluster_size 12 --region_min_samples 0 --region_cluster_selection_method leaf --region_max_region_share 0.30 --region_disable_mid_epoch_topology --region_split_range_fraction 0.20 --region_max_variance_splits_per_epoch 1 --region_split_min_effective_evidence 200 --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 --failure_summary_n_slots 1 --failure_summary_force_recall --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"

LOG_BASE="$MEMRL/logs/aistudio_bcb_regionfs_forced_resume304_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=0 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"

cd "$ROOT"
bash submit.sh
