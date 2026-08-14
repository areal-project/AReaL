#!/bin/bash
#===============================================================================
# Cross-Benchmark 实验环境配置文件
#
# 使用方式: source scripts/cross_benchmark_env.sh
#===============================================================================

# LiteLLM 服务配置
export LLM_API_KEY="sk-placeholder"  # LiteLLM 通常不需要真实 key，但需要有值
export LLM_BASE_URL="http://127.0.0.1:4000"
export LLM_MODEL="gpt-4o-2024-11-20"

# Embedding 配置
export EMBEDDING_API_KEY="${LLM_API_KEY}"
export EMBEDDING_BASE_URL="${LLM_BASE_URL}"
export EMBEDDING_MODEL="text-embedding-3-small"

# 实验默认配置
export NUM_SECTIONS="${NUM_SECTIONS:-10}"      # 训练 epochs
export BATCH_SIZE="${BATCH_SIZE:-5}"           # 批大小
export RANDOM_SEED="${RANDOM_SEED:-42}"        # 随机种子

# SLURM 配置 (如果使用集群)
export PARTITION="${PARTITION:-all}"
export GPUS="${GPUS:-1}"
export CPUS="${CPUS:-8}"
export MEM="${MEM:-32G}"
export TIME="${TIME:-24:00:00}"

echo "[INFO] Cross-Benchmark 环境变量已加载"
echo "  LLM_BASE_URL: ${LLM_BASE_URL}"
echo "  LLM_MODEL: ${LLM_MODEL}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
