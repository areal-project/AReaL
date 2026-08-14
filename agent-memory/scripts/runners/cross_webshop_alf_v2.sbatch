#!/bin/bash
#SBATCH --job-name=cross_webshop_alf_v2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v2_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v2_%j.err

set -ex

echo "[INFO] Starting job on $(hostname)"
cd /storage/openpsi/users/yl/agent-memory/MemRL

# Clean up pycache before running
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "set -ex && cd /storage/openpsi/users/yl/agent-memory/MemRL && pip install 'litellm[proxy]' -q 2>&1 | tail -3 && bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/runners/run_cross_experiment.sh webshop alf cross_webshop_to_alf_v2"
