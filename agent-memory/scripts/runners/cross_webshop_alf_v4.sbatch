#!/bin/bash
#SBATCH --job-name=cross_webshop_alf_v4
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v4_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v4_%j.err

set -ex

echo "[INFO] Starting job on $(hostname)"
echo "[INFO] Working directory: $(pwd)"

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "set -ex && export PYTHONDONTWRITEBYTECODE=1 && pip install 'litellm[proxy]' -q 2>&1 | tail -5 && bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/runners/run_cross_experiment.sh webshop alf cross_webshop_to_alf_v4"
