#!/bin/bash
# HLE Region+FS with gemini-3.1-pro-preview (API-only, no GPU needed)
# Rate-limited to simulate bs=32 without overwhelming the API
#
# Usage: bash scripts/run_hle_region_gemini31.sh
# Or:    sbatch scripts/sbatch_hle_region_gemini31.sh (for slurm)

set -e

cd /storage/openpsi/users/yl/agent-memory/MemRL

# Rate limiter: space out requests to avoid 429 at bs=32
export MEMRL_LLM_MIN_INTERVAL=0.3
export MEMRL_EMBED_MIN_INTERVAL=0.5

# Install dependencies
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true

echo "=========================================="
echo "HLE Region+FS: gemini-3.1-pro-preview via MatrixLLM API"
echo "Start: $(date)"
echo "=========================================="

python3 run/run_hle_region.py \
    --config configs/rl_hle_config.region_gemini31.yaml \
    --train data/hle/hle_test.parquet \
    --text_only \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url "https://matrixllm.alipay.com/v1/" \
    --judge_api_key "sk-43dd5f664179406d92fec42a9364f8a5" \
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

echo "=========================================="
echo "Done. End: $(date)"
echo "=========================================="
