#!/bin/bash
#SBATCH --job-name=yl-llbos-memrl-gpt4o
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=logs/llb_os_memrl_gpt4o_%j.log
#SBATCH --error=logs/llb_os_memrl_gpt4o_%j.log

# MemRL - LLB OS Interaction with gpt-4o (contrast experiment)
# Purpose: verify whether model capability is the ceiling for MemRL gains

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "MemRL - LLB OS (gpt-4o) Experiment"
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
export MEMRL_DB_BACKEND=local
cd ${PROJECT_DIR}

echo '[INFO] Installing dependencies...'
pip install -e . --quiet || echo 'Warning: pip install -e . failed'
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard docker mysql-connector-python --quiet || echo 'Warning: pip install deps failed'

echo 'Python: ' && which python
echo '[INFO] Running LLB OS MemRL (gpt-4o) experiment...'

python run/run_llb.py --config configs/rl_llb_os_gpt4o.yaml

echo '[INFO] LLB OS (gpt-4o) experiment completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
