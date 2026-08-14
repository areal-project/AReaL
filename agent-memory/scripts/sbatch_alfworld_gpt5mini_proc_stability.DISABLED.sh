#!/bin/bash
#SBATCH --job-name=yl-alf-gpt5mini-proc-stable
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/slurm_alf_gpt5mini_proc_stability_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/slurm_alf_gpt5mini_proc_stability_%j.out
# Standard-protocol GPT-5 mini Region+FS run, isolated output/checkpoint identity.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
RUNNER_IMG=/storage/openpsi/images/areal-latest.sif
echo "[SLURM] job=${SLURM_JOB_ID:-unknown} host=${SLURMD_NODENAME:-unknown} start=$(date -Is)"
singularity exec --no-home --writable-tmpfs --bind /storage:/storage "$RUNNER_IMG" \
  bash "$MEMRL_DIR/scripts/run_alf_gpt5mini_proc_stability.sh"
