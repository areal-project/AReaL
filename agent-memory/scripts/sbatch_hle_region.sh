#!/bin/bash
#SBATCH --job-name=yl-hle-region
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/hle_region_%j.log
#SBATCH --error=logs/hle_region_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "HLE Region: gemini-3.1-pro (MatrixLLM) + Additive Gating"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Run HLE Region experiment ---
echo ''
echo '=========================================='
echo '[HLE-REGION] Starting HLE MemRL with Region Transfer'
echo 'Config: gemini-3.1-pro, additive gating, 10 sections'
echo '=========================================='

python run/run_hle_region.py \
    --config configs/rl_hle_config.region.yaml \
    --train data/hle/hle_test.parquet \
    --region_gating_mode additive \
    --region_retrieve_mode global \
    --k_global 30 \
    --k_local 10 \
    --shrinkage_top_n 3 \
    --shrinkage_lambda_max 0.6 \
    --region_utility_mode ema \
    --region_smoothing_C 0.5 \
    --region_cluster_init_step 500 \
    --region_merge_interval 400 \
    --explore_schedule '0,4,3,2,2,1,1,1,1,0'

HLE_EXIT=\$?
echo \"[INFO] HLE Region exited with code: \$HLE_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
