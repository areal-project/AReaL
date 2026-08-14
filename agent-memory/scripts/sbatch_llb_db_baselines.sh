#!/bin/bash
# LLB DB Baselines - Submit via sbatch
# Usage: bash scripts/sbatch_llb_db_baselines.sh [baseline]
# Options: rag | memp | passk | reflection | memrl | all
#
# Examples:
#   bash scripts/sbatch_llb_db_baselines.sh passk
#   bash scripts/sbatch_llb_db_baselines.sh all

set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NODE="slurmd-3"

BASELINE="${1:-all}"

submit_job() {
    local NAME="$1"
    local CONFIG="$2"
    local EXTRA_ARGS="${3:-}"
    local JOB_NAME="yl-llb-db-${NAME}"
    local LOG_FILE="${MEMRL_DIR}/logs/llb_db_${NAME}_${TIMESTAMP}.log"

    echo "Submitting: ${NAME} (config: ${CONFIG})"

    sbatch --job-name=${JOB_NAME} \
        --output=${LOG_FILE} \
        --error=${LOG_FILE} \
        --ntasks=1 \
        --gres=gpu:0 \
        --cpus-per-task=8 \
        --mem=64G \
        --nodelist=${NODE} \
        --chdir=${MEMRL_DIR} \
        --wrap="singularity exec --nv --no-home --writable-tmpfs \
            --bind /storage:/storage \
            ${SINGULARITY_IMG} \
            bash -c '
cd ${MEMRL_DIR}
export HF_ENDPOINT=\"https://hf-mirror.com\"
export HF_HOME=\"/tmp/huggingface\"
pip install -e . --quiet 2>/dev/null
pip install memoryos memos mem0ai chonkie==1.2.1 tensorboard docker mysql-connector-python --quiet 2>/dev/null
apt-get update -qq && apt-get install -y -qq mariadb-server >/dev/null 2>&1 || true
echo \"==========================================\"
echo \"LLB DB Baseline: ${NAME}\"
echo \"Job: \${SLURM_JOB_ID}\"
echo \"Node: \$(hostname)\"
echo \"Start: \$(date)\"
echo \"Config: ${CONFIG}\"
echo \"==========================================\"
python run/run_llb.py --config ${CONFIG} ${EXTRA_ARGS}
'"

    echo "  -> Log: ${LOG_FILE}"
}

case "${BASELINE}" in
    rag)
        submit_job "rag" "configs/rl_llb_db_rag.yaml"
        ;;
    memp)
        submit_job "memp" "configs/rl_llb_db_memp.yaml"
        ;;
    passk)
        submit_job "passk" "configs/rl_llb_db_passk.yaml"
        ;;
    reflection)
        submit_job "reflection" "configs/rl_llb_db_reflection.yaml"
        ;;
    memrl)
        submit_job "memrl" "configs/rl_llb_db_config.local.yaml"
        ;;
    selfrag)
        submit_job "selfrag" "configs/rl_llb_db_selfrag.yaml" "--self_rag"
        ;;
    mem0)
        submit_job "mem0" "configs/rl_llb_db_mem0.yaml" "--mem0"
        ;;
    region_fs)
        submit_job "region_fs" "configs/rl_llb_db_region_fs.yaml" \
            "--region --region_gating_mode additive --failure_summary_n_slots 2 --no_z_norm --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --explore_schedule 0,2,2,1,1,1,1,0,0,0"
        ;;
    all)
        submit_job "rag" "configs/rl_llb_db_rag.yaml"
        submit_job "memp" "configs/rl_llb_db_memp.yaml"
        submit_job "passk" "configs/rl_llb_db_passk.yaml"
        submit_job "reflection" "configs/rl_llb_db_reflection.yaml"
        submit_job "selfrag" "configs/rl_llb_db_selfrag.yaml" "--self_rag"
        submit_job "mem0" "configs/rl_llb_db_mem0.yaml" "--mem0"
        submit_job "memrl" "configs/rl_llb_db_config.local.yaml"
        submit_job "region_fs" "configs/rl_llb_db_region_fs.yaml" \
            "--region --region_gating_mode additive --failure_summary_n_slots 2 --no_z_norm --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --explore_schedule 0,2,2,1,1,1,1,0,0,0"
        ;;
    *)
        echo "Usage: $0 [rag|memp|passk|reflection|selfrag|mem0|memrl|region_fs|all]"
        exit 1
        ;;
esac

echo "Done!"
