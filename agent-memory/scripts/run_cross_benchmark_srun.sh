#!/bin/bash
#===============================================================================
# Cross-Benchmark 实验启动脚本 (Singularity + SLURM)
#
# 使用 srun 提交到计算节点，通过 Singularity 容器运行
#
# 使用方式:
#   ./scripts/run_cross_benchmark_srun.sh [source] [targets...]
#
# 示例:
#   ./scripts/run_cross_benchmark_srun.sh llb hle      # LLB训练 -> HLE测试
#   ./scripts/run_cross_benchmark_srun.sh llb hle bcb  # LLB训练 -> HLE和BCB测试
#===============================================================================

set -e

# 项目路径
PROJECT_ROOT="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMAGE="/storage/openpsi/images/areal-latest.sif"

# 解析参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_benchmark> <target1> [target2] ..."
    echo "  Benchmarks: llb, hle, bcb, alf"
    echo ""
    echo "Examples:"
    echo "  $0 llb hle       # Train on LLB, test on HLE"
    echo "  $0 llb hle bcb   # Train on LLB, test on HLE and BCB"
    exit 1
fi

SOURCE_BENCHMARK=$1
shift
TARGET_BENCHMARKS="$*"
TARGETS_ARRAY=($TARGET_BENCHMARKS)

# 实验配置
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross_${SOURCE_BENCHMARK}_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-5}"

# API 配置 (LiteLLM)
LLM_API_KEY="${LLM_API_KEY:-sk-placeholder}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:4000}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-2024-11-20}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"

# SLURM 配置
PARTITION="${PARTITION:-all}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-24:00:00}"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs"

echo "=========================================="
echo "Cross-Benchmark Experiment (Singularity)"
echo "=========================================="
echo "Source: ${SOURCE_BENCHMARK}"
echo "Targets: ${TARGET_BENCHMARKS}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Image: ${SINGULARITY_IMAGE}"
echo "=========================================="

# 构建 Python 命令 (先安装依赖，再运行实验)
PYTHON_CMD="cd ${PROJECT_ROOT} && \
pip install -e . -q 2>/dev/null && \
pip install -r requirements.txt -q 2>/dev/null && \
python scripts/run_cross_benchmark_experiment.py \
    --source ${SOURCE_BENCHMARK} \
    --targets ${TARGET_BENCHMARKS} \
    --api_key '${LLM_API_KEY}' \
    --base_url '${LLM_BASE_URL}' \
    --model '${LLM_MODEL}' \
    --embedding_model '${EMBEDDING_MODEL}' \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --name '${EXPERIMENT_NAME}' \
    --mode local"

# 使用 srun 提交到计算节点
echo ""
echo "[INFO] 提交到计算节点..."
echo "[INFO] Partition: ${PARTITION}, GPU: ${GPUS}, CPU: ${CPUS}, MEM: ${MEM}"
echo ""

srun --partition="${PARTITION}" \
     --gres="gpu:${GPUS}" \
     --cpus-per-task="${CPUS}" \
     --mem="${MEM}" \
     --time="${TIME}" \
     --mpi=none \
     --job-name="${EXPERIMENT_NAME}" \
     --output="${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}_%j.out" \
     --error="${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}_%j.err" \
     singularity exec --nv \
         --bind /storage:/storage \
         "${SINGULARITY_IMAGE}" \
         bash -c "${PYTHON_CMD}"

echo ""
echo "[INFO] 实验完成!"
echo "[INFO] 日志: ${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}_*.out"
echo "[INFO] 结果: ${PROJECT_ROOT}/results/"
