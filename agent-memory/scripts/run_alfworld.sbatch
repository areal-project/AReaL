#!/bin/bash
#SBATCH --job-name=yl-memrl-alf
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/alfworld_slurm_%j.log
#SBATCH --error=logs/alfworld_slurm_%j.log

export PYTHONUSERBASE=/storage/openpsi/users/yl/agent-memory/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /storage/openpsi/users/yl/agent-memory/MemRL

singularity exec \
  --bind /storage/openpsi/users/yl:/storage/openpsi/users/yl \
  /storage/openpsi/images/areal-latest.sif \
  bash -c "
export PYTHONUSERBASE=/storage/openpsi/users/yl/agent-memory/.local
export PATH=\$PYTHONUSERBASE/bin:\$PATH
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:\$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /storage/openpsi/users/yl/agent-memory/MemRL
python run/run_alfworld.py --config configs/rl_alf_config.local.yaml
"
