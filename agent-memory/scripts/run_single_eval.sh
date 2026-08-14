#!/bin/bash
#===============================================================================
# 单个评估任务脚本 - 用于SLURM依赖调度
#===============================================================================

set -e

TARGET_BENCHMARK=$1
EXPERIMENT_NAME=$2

PROJECT_ROOT="/storage/openpsi/users/yl/agent-memory/MemRL"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
RESULTS_DIR="${PROJECT_ROOT}/results"

# 找到最新的checkpoint
find_latest_checkpoint() {
    local exp_pattern=$1
    local checkpoint_dir=""

    # 查找最新的实验目录
    local latest_exp=$(ls -td "${RESULTS_DIR}/"*"/exp_${exp_pattern}"*/ 2>/dev/null | head -1)

    if [ -n "${latest_exp}" ] && [ -d "${latest_exp}/snapshot" ]; then
        checkpoint_dir=$(ls -td "${latest_exp}/snapshot/"*/ 2>/dev/null | head -1)
    fi

    echo "${checkpoint_dir}"
}

# Source环境变量
source "${SCRIPT_DIR}/cross_benchmark_experiment.sh" 2>/dev/null || true

# 找checkpoint
CHECKPOINT_PATH=$(find_latest_checkpoint "${EXPERIMENT_NAME}")

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "[ERROR] 未找到checkpoint"
    exit 1
fi

echo "[INFO] 使用checkpoint: ${CHECKPOINT_PATH}"

# 生成评估配置
EVAL_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${TARGET_BENCHMARK}_eval.yaml"

# 这里需要调用生成配置的函数，或者使用Python脚本
python "${SCRIPT_DIR}/generate_eval_config.py" \
    --benchmark "${TARGET_BENCHMARK}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output "${EVAL_CONFIG}"

# 运行评估
cd "${PROJECT_ROOT}"
python "run/run_${TARGET_BENCHMARK}.py" --config "${EVAL_CONFIG}"

echo "[INFO] 评估完成: ${TARGET_BENCHMARK}"
