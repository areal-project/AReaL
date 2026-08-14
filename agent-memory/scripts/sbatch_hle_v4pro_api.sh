#!/bin/bash
#SBATCH --job-name=yl-hle-v4pro-api
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/hle_v4pro_api_%j.log
#SBATCH --error=logs/hle_v4pro_api_%j.log

# API-based HLE baselines (DeepSeek-V4-Pro via MatrixLLM). No GPU needed.
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "HLE Baselines via DeepSeek-V4-Pro API (no-memory + MemRL), bs=6"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

# --- Baseline 1: no-memory (pass@1) ---
echo ''
echo '[BASELINE 1/2] HLE no-memory (pass@1) via V4-Pro API'
python run/run_hle.py \
    --config configs/rl_hle_config.nomem_v4pro_api.yaml \
    --train data/hle/hle_test.parquet \
    --text_only \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
NOMEM_EXIT=\$?
echo \"[INFO] no-memory baseline exited: \$NOMEM_EXIT\"

# --- Baseline 2: MemRL (memory, no region) ---
echo ''
echo '[BASELINE 2/2] HLE MemRL via V4-Pro API'
python run/run_hle.py \
    --config configs/rl_hle_config.memrl_v4pro_api.yaml \
    --train data/hle/hle_test.parquet \
    --text_only \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
MEMRL_EXIT=\$?
echo \"[INFO] MemRL baseline exited: \$MEMRL_EXIT\"
echo \"[INFO] no-mem: \$NOMEM_EXIT | MemRL: \$MEMRL_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
