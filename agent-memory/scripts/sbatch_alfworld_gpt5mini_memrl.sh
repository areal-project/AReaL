#!/bin/bash
#SBATCH --job-name=yl-alf-gpt5mini-memrl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/alf_gpt5mini_memrl_%j.log
#SBATCH --error=logs/alf_gpt5mini_memrl_%j.log

# ALFWorld MemRL via gpt-5-mini (matrixllm API). NO GPU — LLM + embedding both API.
# Goal: reproduce paper ~94.9% on full train set (in-distribution, 10 sections).
# OpenAILLM auto-handles gpt-5 reasoning params (max_completion_tokens, no temperature).

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "ALFWorld MemRL — gpt-5-mini API (no GPU), bs=32, full train set"
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
echo '[RUN] ALFWorld MemRL gpt-5-mini (in-distribution, full train, 10 sections)'
python run/run_alfworld.py \
    --config configs/rl_alf_config.gpt5mini_memrl.yaml
echo \"[alf gpt5mini memrl] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
