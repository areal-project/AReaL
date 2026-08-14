#!/bin/bash
#SBATCH --job-name=yl-llb-os-injdbg
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/llb_os_injdbg_%j.log
#SBATCH --error=logs/llb_os_injdbg_%j.log

# Debug: verify LLB OS MemRL memory injection / success-bucketing.
# Zero API calls, zero side effects. Runs inside Singularity container.

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "LLB OS injection debug"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "=========================================="

mkdir -p ${PROJECT_DIR}/logs

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
export MEMRL_OS_BACKEND=local
cd ${PROJECT_DIR}

echo '[INFO] Installing memos deps (quiet)...'
pip install -e . --quiet || echo 'Warning: pip install -e . failed'
pip install memoryos memos mem0ai 'chonkie==1.2.1' --quiet || echo 'Warning: pip install deps failed'

echo 'Python: ' && which python
echo '[INFO] Running injection debug...'
python scripts/debug_llb_os_injection.py
"

echo "=========================================="
echo "End: $(date)  Exit: $?"
echo "=========================================="
