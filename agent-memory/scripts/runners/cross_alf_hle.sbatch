#!/bin/bash
#SBATCH --job-name=cross_alf_hle_v4
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_alf_hle_v4_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_alf_hle_v4_%j.err

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/runners/run_cross_experiment.sh alf hle cross_alf_to_hle_v4
