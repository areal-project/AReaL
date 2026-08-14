#!/bin/bash
#SBATCH --job-name=yl-hle-nomem-g35f
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/hle_nomem_gemini35flash_%j.log
#SBATCH --error=logs/hle_nomem_gemini35flash_%j.log

# HLE no-memory baseline with gemini-3.5-flash (API-only, no GPU)

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "HLE No-Memory Baseline: gemini-3.5-flash via MatrixLLM API"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_MIN_INTERVAL=1.0
export MEMRL_REASONING_EFFORT=high
python3 run/run_hle.py \
    --config configs/rl_hle_config.nomem_gemini35flash.yaml \
    --train data/hle/hle_test.parquet \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
"
EXIT_CODE=$?

echo "[INFO] Done. Exit: $EXIT_CODE | End: $(date)"
