#!/bin/bash
#SBATCH --job-name=yl-memrl-llb-os
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/llb_os_%j.log
#SBATCH --error=logs/llb_os_%j.log

# MemRL - Lifelong Agent Bench (OS Interaction) experiment
# Uses LocalOSContainer (subprocess) - no Docker/Apptainer nesting needed

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "MemRL - Lifelong Agent Bench (OS) Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p ${PROJECT_DIR}/logs

singularity exec --nv \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=local
cd ${PROJECT_DIR}

pip install docker mysql-connector-python --quiet 2>/dev/null || true

echo 'Python: ' && which python
echo 'Running LLB OS experiment...'

python run/run_llb.py --config configs/rl_llb_config.local.yaml
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
