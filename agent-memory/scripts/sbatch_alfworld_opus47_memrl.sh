#!/bin/bash
#SBATCH --job-name=yl-alf-opus47-memrl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/alf_opus47_memrl_%j.log
#SBATCH --error=logs/alf_opus47_memrl_%j.log

# ALFWorld MemRL via claude-opus-4-7 (matrixllm API). NO GPU.
# Send-rate throttle (MEMRL_LLM_MIN_INTERVAL) avoids gateway 429s — verified
# to cut 429s from thousands to single digits at 0.15s spacing.
# Target: opus-4-7 no-mem=82.86% (already measured), reproduce paper-style
# memrl gain (paper gpt-5-mini transfer 83.6 -> 97.9).
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "ALFWorld MemRL — claude-opus-4-7 API (throttled, no GPU), full train 10 sections"
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
export MEMRL_LLM_MIN_INTERVAL=${MEMRL_LLM_MIN_INTERVAL:-0.15}
echo \"[INFO] MEMRL_LLM_MIN_INTERVAL=\$MEMRL_LLM_MIN_INTERVAL\"

echo ''
echo '[RUN] ALFWorld MemRL opus-4-7 (in-distribution, full train, 10 sections)'
python run/run_alfworld.py --config configs/rl_alf_config.opus47_memrl.yaml
echo \"[alf opus47 memrl] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
