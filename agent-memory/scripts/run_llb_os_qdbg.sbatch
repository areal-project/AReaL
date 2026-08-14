#!/bin/bash
#SBATCH --job-name=yl-llb-os-qdbg
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/llb_os_qdbg_%j.log
#SBATCH --error=logs/llb_os_qdbg_%j.log

# Debug: Q-value spread across OS MemRL snapshots. Pure JSON, no deps, no API.
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "LLB OS Q-value debug | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date)"
mkdir -p ${PROJECT_DIR}/logs

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "cd ${PROJECT_DIR} && python scripts/debug_llb_os_qvalues.py"

echo "Exit: $?  $(date)"
