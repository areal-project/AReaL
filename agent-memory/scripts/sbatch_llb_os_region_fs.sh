#!/bin/bash
# MemRL LLB OS Region+FS Experiment - Submit via sbatch
# Usage: bash scripts/sbatch_llb_os_region_fs.sh

set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="yl-llb-region-fs"
CONFIG_FILE="configs/rl_llb_os_config.local.yaml"
LOG_FILE="${MEMRL_DIR}/logs/llb_os_region_fs_${TIMESTAMP}.log"

sbatch --job-name=${JOB_NAME} \
    --output=${LOG_FILE} \
    --error=${LOG_FILE} \
    --ntasks=1 \
    --gres=gpu:0 \
    --cpus-per-task=8 \
    --mem=64G \
    --nodelist=slurmd-3 \
    --chdir=${MEMRL_DIR} \
    --wrap="singularity exec --nv --no-home --writable-tmpfs \
        --bind /storage:/storage \
        ${SINGULARITY_IMG} \
        bash -c '
cd ${MEMRL_DIR}
export HF_ENDPOINT=\"https://hf-mirror.com\"
export HF_HOME=\"/tmp/huggingface\"
pip install -e . --quiet 2>/dev/null
pip install memoryos memos mem0ai chonkie==1.2.1 tensorboard docker --quiet 2>/dev/null
echo \"==========================================\"
echo \"MemRL LLB OS Region+FS Experiment\"
echo \"Job: \${SLURM_JOB_ID}\"
echo \"Node: \$(hostname)\"
echo \"Start: \$(date)\"
echo \"Config: ${CONFIG_FILE}\"
echo \"==========================================\"
python run/run_llb.py --config ${CONFIG_FILE} \
    --region \
    --failure_summary_n_slots 2 \
    --failure_summary_k 10
'"

echo "Submitted! Log file: ${LOG_FILE}"
