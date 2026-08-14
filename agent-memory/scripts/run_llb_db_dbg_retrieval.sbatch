#!/bin/bash
#SBATCH --job-name=yl-llb-db-dbg-retr
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/llb_db_dbg_retr_%j.log
#SBATCH --error=logs/llb_db_dbg_retr_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "LLB DB — Retrieval content debug"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p ${PROJECT_DIR}/logs

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
export MEMRL_OS_BACKEND=local
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
cd ${PROJECT_DIR}

echo '[INFO] Installing Python deps (no MariaDB needed for this debug)...'
pip install -e . --quiet || echo 'Warning: pip install -e . failed'
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet || echo 'Warning: pip install deps failed'

echo '[INFO] Running retrieval debug...'
python scripts/debug_llb_db_retrieval.py
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
