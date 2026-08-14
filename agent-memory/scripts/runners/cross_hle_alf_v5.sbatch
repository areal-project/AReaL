#!/bin/bash
#SBATCH --job-name=cross_hle_alf_v5
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_hle_alf_v5_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_hle_alf_v5_%j.err

set -ex

echo "[INFO] Starting job on $(hostname)"
echo "[INFO] Working directory: $(pwd)"

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "set -ex && pip install -e . -q 2>&1 | tail -5 && pip install 'litellm[proxy]' -q 2>&1 | tail -5 && bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/runners/run_cross_experiment.sh hle alf cross_hle_to_alf_v5"
