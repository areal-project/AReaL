#!/bin/bash
set -euo pipefail
METHOD="${1:?usage: $0 memp|rag|memrl_eval|leaf_eval}"
ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL=$ROOT/MemRL
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench
RUNNER=$MEMRL/scripts/aistudio_bcb_runner_strict.sh
STAMP=$(date +%Y%m%d_%H%M%S)
export WORKER_NUM="${WORKER_NUM:-1}"
export MASTER_GPU_NUM="${MASTER_GPU_NUM:-1}"
export SUBMIT_TEMPLATE="${SUBMIT_TEMPLATE:-$MEMRL/scripts/submit_template_bcb_strict.py}"
export BCB_EPOCHS=10
export BCB_NO_AUTORESUME=1
export BCB_ENTRYPOINT=run/run_bcb.py
case "$METHOD" in
  memp)
    export JOB_NAME="yl-bcb-memp-complete-$STAMP"
    export BCB_CONFIG=configs/rl_bcb_config.memp_local.yaml
    export BCB_OUTPUT_DIR=$OUT/deepseek_v3_memp
    SNAP=$OUT/deepseek_v3_memp/bigcodebench_eval/instruct_full/memory/20260710_032817_deepseek-ai_DeepSeek-V3_rl-off/epoch9/snapshot/9
    export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 9 --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
    ;;
  rag)
    export JOB_NAME="yl-bcb-rag-complete-$STAMP"
    export BCB_CONFIG=configs/rl_bcb_config.rag_local.yaml
    export BCB_OUTPUT_DIR=$OUT/deepseek_v3_rag
    SNAP=$OUT/deepseek_v3_rag/bigcodebench_eval/instruct_full/memory/20260712_071437_deepseek-ai_DeepSeek-V3_rl-off/epoch8/snapshot/8
    export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 8 --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
    ;;
  memrl_eval)
    export JOB_NAME="yl-bcb-memrl-e10-multieval-$STAMP"
    export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local.yaml
    export BCB_OUTPUT_DIR=$OUT/deepseek_v3_local_memrl_multieval
    export BCB_ENTRYPOINT=scripts/run_bcb_multi_eval_only.py
    SNAP=$OUT/deepseek_v3_local_memrl/bigcodebench_eval/instruct_full/memory/20260619_165450_deepseek-ai_DeepSeek-V3_rl-on/epoch10/snapshot/10
    export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 10 --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
    ;;
  leaf_eval)
    export JOB_NAME="yl-bcb-leaf-e10-multieval-$STAMP"
    export BCB_CONFIG=configs/rl_bcb_config.deepseek_v3_local_region.yaml
    export BCB_OUTPUT_DIR=$OUT/deepseek_v3_local_region_fs_leaf_multieval
    export BCB_ENTRYPOINT=scripts/run_bcb_region_multi_eval_only.py
    SNAP=$OUT/deepseek_v3_local_region_fs_leaf/bigcodebench_eval/instruct_full/region/20260705_163036_deepseek-ai_DeepSeek-V3_region/epoch10/snapshot/10
    export BCB_EXTRA_ARGS="--resume_from $SNAP --resume_epoch 10 --eval_only --task_cluster_k 0 --region_gating_mode additive --region_utility_mode beta --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 12 --region_min_samples 0 --region_cluster_selection_method leaf --region_max_region_share 0.30 --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 --failure_summary_n_slots 1"
    ;;
  *) echo "unknown method $METHOD"; exit 2;;
esac
LOG_BASE=$MEMRL/logs/aistudio_${METHOD}_${STAMP}
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_ENTRYPOINT=\"${BCB_ENTRYPOINT}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=1 bash ${RUNNER}'"
cd "$ROOT"
bash submit.sh
