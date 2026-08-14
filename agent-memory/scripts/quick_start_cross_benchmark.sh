#!/bin/bash
#===============================================================================
# 快速启动脚本 - 使用srun在计算节点运行跨benchmark实验
#
# 使用方式:
#   ./quick_start_cross_benchmark.sh [source_benchmark] [target_benchmarks...]
#
# 示例:
#   ./quick_start_cross_benchmark.sh llb hle bcb     # 在LLB上训练，测试HLE和BCB
#   ./quick_start_cross_benchmark.sh bcb llb alf     # 在BCB上训练，测试LLB和ALF
#
# 环境变量 (必须设置):
#   LLM_API_KEY       - API密钥
#   LLM_BASE_URL      - API地址 (默认: https://api.openai.com/v1)
#   LLM_MODEL         - 模型名称 (默认: gpt-4o-mini)
#===============================================================================

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_benchmark> <target_benchmark1> [target_benchmark2] ..."
    echo "  source_benchmark: llb, bcb, alf, hle"
    echo "  target_benchmark: llb, bcb, alf, hle"
    echo ""
    echo "Examples:"
    echo "  $0 llb hle bcb    # Train on LLB, test on HLE and BCB"
    echo "  $0 bcb llb        # Train on BCB, test on LLB"
    exit 1
fi

# 检查API密钥
if [ -z "${LLM_API_KEY}" ]; then
    echo "[ERROR] LLM_API_KEY 环境变量未设置"
    echo "请设置: export LLM_API_KEY=your-api-key"
    exit 1
fi

SOURCE_BENCHMARK=$1
shift
TARGET_BENCHMARKS="$*"

# 项目路径
PROJECT_ROOT="/storage/openpsi/users/yl/agent-memory/MemRL"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="cross_${SOURCE_BENCHMARK}_${TIMESTAMP}"

echo "=========================================="
echo "Cross-Benchmark Memory Transfer Experiment"
echo "=========================================="
echo "源Benchmark: ${SOURCE_BENCHMARK}"
echo "目标Benchmarks: ${TARGET_BENCHMARKS}"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "=========================================="

# SLURM配置
PARTITION="${PARTITION:-all}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-24:00:00}"

# 设置环境变量
export SOURCE_BENCHMARK
export TARGET_BENCHMARKS
export EXPERIMENT_NAME
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:4000}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-2024-11-20}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-$LLM_API_KEY}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-$LLM_BASE_URL}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
export NUM_SECTIONS="${NUM_SECTIONS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-5}"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs"

echo ""
echo "[INFO] 使用srun提交作业到计算节点..."
echo "[INFO] 分区: ${PARTITION}, GPU: ${GPUS}, CPU: ${CPUS}, 内存: ${MEM}"
echo ""

# 使用srun运行主脚本
srun --partition="${PARTITION}" \
     --gres="gpu:${GPUS}" \
     --cpus-per-task="${CPUS}" \
     --mem="${MEM}" \
     --time="${TIME}" \
     --job-name="${EXPERIMENT_NAME}" \
     --output="${PROJECT_ROOT}/logs/srun_${EXPERIMENT_NAME}_%j.out" \
     --error="${PROJECT_ROOT}/logs/srun_${EXPERIMENT_NAME}_%j.err" \
     bash "${PROJECT_ROOT}/scripts/cross_benchmark_experiment.sh" local

echo ""
echo "[INFO] 实验完成!"
echo "[INFO] 日志目录: ${PROJECT_ROOT}/logs/"
echo "[INFO] 结果目录: ${PROJECT_ROOT}/results/"
