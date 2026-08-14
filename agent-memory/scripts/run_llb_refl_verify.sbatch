#!/bin/bash
#SBATCH --job-name=yl-llb-refl-verify
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/llb_refl_verify_%j.log
#SBATCH --error=logs/llb_refl_verify_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "LLB reflection pipeline verify"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Start: $(date)"
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

echo '[INFO] Installing deps...'
pip install -e . --quiet || echo 'Warning: pip install -e . failed'
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet || echo 'Warning: pip install deps failed'

echo '[INFO] Running reflection verify...'
python scripts/verify_llb_reflection.py
"

echo "=========================================="
echo "End: $(date)  Exit: $?"
echo "=========================================="
