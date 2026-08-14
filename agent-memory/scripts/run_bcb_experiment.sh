#!/bin/bash
#===============================================================================
# BCB Cross-Benchmark Experiment Runner
#
# 使用方式:
#   export LLM_API_KEY="your-api-key"
#   export LLM_BASE_URL="your-base-url"   # 可选
#   export LLM_MODEL="gpt-4o-mini"        # 可选
#   bash scripts/run_bcb_experiment.sh
#
# 注意: 当前环境网络无法访问HuggingFace，HLE数据需要手动下载到 data/hle/hle_test.parquet
#===============================================================================

set -e

PROJECT_ROOT="/storage/openpsi/users/yl/agent-memory/MemRL"
cd "${PROJECT_ROOT}"

# 检查API密钥
if [ -z "${LLM_API_KEY}" ]; then
    echo "[ERROR] LLM_API_KEY 环境变量未设置"
    echo "请设置: export LLM_API_KEY=your-api-key"
    exit 1
fi

# 默认配置
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.openai.com/v1}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-$LLM_API_KEY}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-$LLM_BASE_URL}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"

# 实验配置
NUM_SECTIONS="${NUM_SECTIONS:-10}"
BATCH_SIZE="${BATCH_SIZE:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="bcb_cross_${TIMESTAMP}"

echo "=========================================="
echo "BCB Cross-Benchmark Experiment"
echo "=========================================="
echo "Model: ${LLM_MODEL}"
echo "Base URL: ${LLM_BASE_URL}"
echo "Epochs: ${NUM_SECTIONS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="

# 方式1: 使用Python脚本（推荐）
echo ""
echo "[INFO] Running BCB experiment via Python script..."
python scripts/run_cross_benchmark_experiment.py \
    --source bcb \
    --targets llb \
    --api_key "${LLM_API_KEY}" \
    --base_url "${LLM_BASE_URL}" \
    --model "${LLM_MODEL}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --name "${EXPERIMENT_NAME}" \
    --epochs "${NUM_SECTIONS}" \
    --batch_size "${BATCH_SIZE}" \
    --mode local

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results: ${PROJECT_ROOT}/results/"
echo "=========================================="
