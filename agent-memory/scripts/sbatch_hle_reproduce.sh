#!/bin/bash
#SBATCH --job-name=yl-hle-reproduce
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/hle_reproduce_%j.log
#SBATCH --error=logs/hle_reproduce_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "HLE gemini-3.1-pro (MatrixLLM) reproduction"
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

# --- Run HLE experiment (10 sections, batch_size=1) ---
echo ''
echo '=========================================='
echo '[HLE] Starting HLE MemRL with gemini-3.1-pro'
echo 'Config: paper params, batch_size=1, 10 sections'
echo '=========================================='

python run/run_hle.py \
    --config configs/rl_hle_config.reproduce.yaml \
    --train data/hle/hle_test.parquet

HLE_EXIT=\$?
echo \"[INFO] HLE exited with code: \$HLE_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
