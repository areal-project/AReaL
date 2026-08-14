#!/bin/bash
# Leaf-v8b topology schedule fix: retain raw direct outcomes before the first Region cluster, then
# route them once through initial top-3 sharpened soft memberships. This is a
# correctness-preserving replacement for legacy Q*q_count pseudo-evidence.
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM=1
export MASTER_GPU_NUM=1
export AIS_MAX_ATTEMPT=1
export SUBMIT_TEMPLATE="$MEMRL/scripts/submit_template_bcb_strict.py"
export JOB_NAME="yl-bcb-leafv8b-cooldown1-$STAMP"
export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local_region.yaml
export BCB_EPOCHS=10
export BCB_OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_region_fs_leaf_v8b_cooldown1
export BCB_ENTRYPOINT=run/run_bcb_region.py
export BCB_NO_AUTORESUME=0
# Intentionally no v4 margin abstention: v5 isolates raw pre-cluster evidence
# backfill while retaining all corrected v3 topology and contract-FS semantics.
export BCB_EXTRA_ARGS="--task_cluster_k 0 --region_gating_mode additive --region_utility_mode beta --region_split_evidence_migration_mode soft_source_conserving --region_precluster_evidence_mode soft_source_backfill --region_precluster_evidence_scale 0.75 --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 12 --region_min_samples 0 --region_cluster_selection_method leaf --region_max_region_share 0.30 --region_topology_cooldown_epochs 1 --region_disable_mid_epoch_topology --region_split_range_fraction 0.20 --region_max_variance_splits_per_epoch 1 --region_split_min_effective_evidence 200 --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 --failure_summary_n_slots 1 --failure_summary_contract_filter --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
LOG_BASE="$MEMRL/logs/aistudio_bcb_leafv8b_cooldown1_$STAMP"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=0 BCB_EVAL_MAX_CONCURRENCY=2 BCB_EVAL_DISABLE_TENSORFLOW=1 bash ${MEMRL}/scripts/aistudio_bcb_runner_strict.sh'"
cd "$ROOT"
bash submit.sh
