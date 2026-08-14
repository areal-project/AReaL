#!/bin/bash
#SBATCH --job-name=yl-hle-reg-gem
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/hle_region_gemini31_%j.log
#SBATCH --error=logs/hle_region_gemini31_%j.log

# HLE Region+FS with gemini-3.1-pro-preview (API-only, no GPU)
# Pure API calls to MatrixLLM, rate-limited bs=32

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "HLE Region+FS: gemini-3.1-pro-preview via MatrixLLM API"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_MIN_INTERVAL=1.0
export MEMRL_UPDATE_MAX_WORKERS=4
export MEMRL_REASONING_EFFORT=high
python3 run/run_hle_region.py \
    --config configs/rl_hle_config.region_gemini31.yaml \
    --train data/hle/hle_test.parquet \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5 \
    --region_gating_mode additive \
    --region_retrieve_mode global \
    --k_global 30 \
    --k_local 10 \
    --shrinkage_top_n 3 \
    --shrinkage_lambda_max 0.6 \
    --region_utility_mode ema \
    --region_smoothing_C 0.5 \
    --region_cluster_init_step 300 \
    --region_merge_interval 200 \
    --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
    --failure_summary_n_slots 2
"
EXIT_CODE=$?

echo "[INFO] Done. Exit: $EXIT_CODE | End: $(date)"
