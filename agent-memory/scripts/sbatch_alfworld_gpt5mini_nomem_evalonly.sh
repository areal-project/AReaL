#!/bin/bash
#SBATCH --job-name=yl-alf-g5-evalonly
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/alf_g5_evalonly_%j.log
#SBATCH --error=logs/alf_g5_evalonly_%j.log

# ALFWorld no-mem EVAL-ONLY (140 valid_seen) via gpt-5-mini API. NO GPU.
# Verifies the paper-aligned prompt (anti-reasoning block removed) lifts no-mem SR.
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "ALFWorld no-mem eval-only (gpt-5-mini, paper-aligned prompt)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing dependencies...'
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

echo ''
echo '[RUN] ALFWorld no-mem eval-only (140 valid_seen)'
python run/run_alfworld.py --config configs/rl_alf_config.gpt5mini_nomem_evalonly.yaml
echo \"[alf g5 evalonly] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
