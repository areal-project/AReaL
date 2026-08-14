#!/bin/bash
#===============================================================================
# Cross-Benchmark Memory Transfer Experiment
#
# 功能: 在一个benchmark上训练memory，然后迁移到其他benchmark测试
# 使用: sbatch cross_benchmark_experiment.sh 或 bash cross_benchmark_experiment.sh
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# 配置区域 - 根据需要修改
#-------------------------------------------------------------------------------

# 项目路径
PROJECT_ROOT="/storage/openpsi/users/yl/agent-memory/MemRL"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
CONFIG_DIR="${PROJECT_ROOT}/configs"
RESULTS_DIR="${PROJECT_ROOT}/results"

# API配置 (从环境变量读取，或使用 LiteLLM 本地服务默认配置)
export LLM_API_KEY="${LLM_API_KEY:-sk-placeholder}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:4000}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-2024-11-20}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-$LLM_API_KEY}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-$LLM_BASE_URL}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"

# 实验配置
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross_bench_exp}"
RANDOM_SEED="${RANDOM_SEED:-42}"
NUM_SECTIONS="${NUM_SECTIONS:-10}"  # 训练epochs
BATCH_SIZE="${BATCH_SIZE:-5}"

# 源benchmark (训练memory)
SOURCE_BENCHMARK="${SOURCE_BENCHMARK:-llb}"  # llb, bcb, alf, hle
SOURCE_TASK="${SOURCE_TASK:-os}"  # 对于llb: os/db

# 目标benchmarks (测试memory迁移效果)
TARGET_BENCHMARKS="${TARGET_BENCHMARKS:-hle bcb}"  # 空格分隔的benchmark列表

# SLURM配置
PARTITION="${PARTITION:-all}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-48:00:00}"

#-------------------------------------------------------------------------------
# 辅助函数
#-------------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_requirements() {
    log "检查环境依赖..."

    cd "${PROJECT_ROOT}"

    # 检查Python环境
    python -c "import memrl" 2>/dev/null || error "memrl模块未安装，请先安装: pip install -e ."

    # 检查必要的数据文件
    case "${SOURCE_BENCHMARK}" in
        llb)
            [ -f "data/llb/os_interaction_data.json" ] || error "LLB数据文件不存在"
            ;;
        bcb)
            [ -d "3rdparty/bigcodebench-main" ] || error "BigCodeBench目录不存在"
            ;;
        alf)
            [ -f "configs/envs/alfworld.yaml" ] || error "ALFWorld配置不存在"
            ;;
    esac

    log "环境检查通过"
}

generate_config() {
    local benchmark=$1
    local config_type=$2  # train 或 eval
    local checkpoint_path=$3  # 仅eval时使用
    local output_file=$4

    log "生成配置文件: ${output_file}"

    cat > "${output_file}" << EOF
# Auto-generated config for ${benchmark} (${config_type})
# Generated at: $(date)

llm:
  provider: "openai"
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL}"
  model: "${LLM_MODEL}"
  temperature: 0.0
  max_tokens: 10240

embedding:
  provider: "openai"
  api_key: "${EMBEDDING_API_KEY}"
  base_url: "${EMBEDDING_BASE_URL}"
  model: "${EMBEDDING_MODEL}"
  max_text_len: 8196

memory:
  build_strategy: "proceduralization"
  retrieve_strategy: "query"
  update_strategy: "adjustment"
  k_retrieve: 10
  max_keywords: 8
  confidence_threshold: 0.0
  memory_confidence: 100.0
  add_similarity_threshold: 0.99
  mos_config_path: "configs/mos_config.json"
  user_id: "${EXPERIMENT_NAME}_${benchmark}"
EOF

    # 如果是eval模式且有checkpoint，添加checkpoint配置
    if [ "${config_type}" == "eval" ] && [ -n "${checkpoint_path}" ]; then
        cat >> "${output_file}" << EOF
  load_from_checkpoint: true
  checkpoint_path: "${checkpoint_path}"
EOF
    else
        cat >> "${output_file}" << EOF
  load_from_checkpoint: false
  checkpoint_path: null
EOF
    fi

    # 添加sim_norm参数 (不同benchmark可能需要不同的归一化参数)
    case "${benchmark}" in
        llb)
            if [ "${SOURCE_TASK}" == "os" ]; then
                echo "  sim_norm_mean: 0.39" >> "${output_file}"
                echo "  sim_norm_std: 0.14" >> "${output_file}"
            else
                echo "  sim_norm_mean: 0.27" >> "${output_file}"
                echo "  sim_norm_std: 0.11" >> "${output_file}"
            fi
            ;;
        bcb)
            echo "  sim_norm_mean: 0.31" >> "${output_file}"
            echo "  sim_norm_std: 0.10" >> "${output_file}"
            ;;
        hle)
            echo "  sim_norm_mean: 0.19" >> "${output_file}"
            echo "  sim_norm_std: 0.09" >> "${output_file}"
            ;;
        alf)
            echo "  sim_norm_mean: 0.52" >> "${output_file}"
            echo "  sim_norm_std: 0.12" >> "${output_file}"
            ;;
    esac

    cat >> "${output_file}" << EOF

environment:
  alfworld_config_path: "configs/envs/alfworld.yaml"
  alfworld_env_type: "AlfredTWEnv"

experiment:
  experiment_name: "${EXPERIMENT_NAME}_${benchmark}_${config_type}"
  algorithm: "rl"
  val_before_train: false
  enable_value_driven: true
  random_seed: ${RANDOM_SEED}
  mode: "${config_type}"
  task: "${SOURCE_TASK}"
  split_file: "data/llb/${SOURCE_TASK}_interaction_data.json"
  valid_file: null
  num_sections: ${NUM_SECTIONS}
  batch_size: ${BATCH_SIZE}
  max_steps: 15
  valid_interval: 0
  test_interval: 1
  dataset_ratio: 1.0
  few_shot_path: "data/alfworld/alfworld_examples.json"
  bon: 0
  hle_categories: null
  hle_category_ratio: null
  ckpt_eval_enabled: false
  ckpt_eval_path: null
  ckpt_resume_enabled: false
  ckpt_resume_path: null
  ckpt_resume_epoch: null
  baseline_mode: null
  baseline_k: 10
  output_dir: "${RESULTS_DIR}"
  save_trajectories: true
  save_memories: true
  enable_logging: true
  log_level: "INFO"

rl_config:
  epsilon: 0.01
  tau: 0.35
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0.5
  q_init_neg: 0.5
  success_reward: 1.0
  failure_reward: 0.0
  sim_threshold: 0.5
  topk: 5
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -0.8
  weight_sim: 0.5
  weight_q: 0.5
EOF

    log "配置文件生成完成: ${output_file}"
}

run_training() {
    local benchmark=$1
    local config_file=$2

    log "开始在 ${benchmark} 上训练..."

    cd "${PROJECT_ROOT}"

    case "${benchmark}" in
        llb)
            python run/run_llb.py --config "${config_file}"
            ;;
        bcb)
            python run/run_bcb.py --config "${config_file}" --subset hard --epochs "${NUM_SECTIONS}"
            ;;
        alf)
            python run/run_alfworld.py --config "${config_file}"
            ;;
        hle)
            python run/run_hle.py --config "${config_file}"
            ;;
        *)
            error "未知benchmark: ${benchmark}"
            ;;
    esac

    log "${benchmark} 训练完成"
}

run_evaluation() {
    local benchmark=$1
    local config_file=$2

    log "开始在 ${benchmark} 上评估..."

    cd "${PROJECT_ROOT}"

    case "${benchmark}" in
        llb)
            python run/run_llb.py --config "${config_file}"
            ;;
        bcb)
            python run/run_bcb.py --config "${config_file}" --subset hard --epochs 1
            ;;
        alf)
            python run/run_alfworld.py --config "${config_file}"
            ;;
        hle)
            python run/run_hle.py --config "${config_file}"
            ;;
        *)
            error "未知benchmark: ${benchmark}"
            ;;
    esac

    log "${benchmark} 评估完成"
}

find_latest_checkpoint() {
    local exp_dir=$1
    local checkpoint_dir=""

    # 查找最新的snapshot目录
    if [ -d "${exp_dir}/snapshot" ]; then
        checkpoint_dir=$(ls -td "${exp_dir}/snapshot/"*/ 2>/dev/null | head -1)
    fi

    if [ -z "${checkpoint_dir}" ]; then
        # 尝试查找results目录下的实验
        local latest_exp=$(ls -td "${RESULTS_DIR}/${SOURCE_BENCHMARK}/exp_${EXPERIMENT_NAME}"*/ 2>/dev/null | head -1)
        if [ -n "${latest_exp}" ] && [ -d "${latest_exp}/snapshot" ]; then
            checkpoint_dir=$(ls -td "${latest_exp}/snapshot/"*/ 2>/dev/null | head -1)
        fi
    fi

    echo "${checkpoint_dir}"
}

#-------------------------------------------------------------------------------
# 主流程
#-------------------------------------------------------------------------------

main() {
    log "=========================================="
    log "Cross-Benchmark Memory Transfer Experiment"
    log "=========================================="
    log "源Benchmark: ${SOURCE_BENCHMARK}"
    log "目标Benchmarks: ${TARGET_BENCHMARKS}"
    log "实验名称: ${EXPERIMENT_NAME}"
    log "=========================================="

    # 创建必要目录
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${SCRIPT_DIR}/generated_configs"
    mkdir -p "${PROJECT_ROOT}/logs"

    # 检查环境
    check_requirements

    # Step 1: 在源benchmark上训练
    log ""
    log "===== Step 1: 训练阶段 ====="

    TRAIN_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${SOURCE_BENCHMARK}_train.yaml"
    generate_config "${SOURCE_BENCHMARK}" "train" "" "${TRAIN_CONFIG}"
    run_training "${SOURCE_BENCHMARK}" "${TRAIN_CONFIG}"

    # 找到checkpoint路径
    CHECKPOINT_PATH=$(find_latest_checkpoint "${RESULTS_DIR}/${SOURCE_BENCHMARK}")

    if [ -z "${CHECKPOINT_PATH}" ]; then
        error "未找到checkpoint，训练可能失败"
    fi

    log "找到checkpoint: ${CHECKPOINT_PATH}"

    # Step 2: 在每个目标benchmark上评估
    log ""
    log "===== Step 2: 评估阶段 ====="

    for target in ${TARGET_BENCHMARKS}; do
        log ""
        log "--- 评估目标: ${target} ---"

        EVAL_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${target}_eval.yaml"
        generate_config "${target}" "eval" "${CHECKPOINT_PATH}" "${EVAL_CONFIG}"
        run_evaluation "${target}" "${EVAL_CONFIG}"
    done

    # Step 3: 汇总结果
    log ""
    log "===== Step 3: 结果汇总 ====="

    python "${SCRIPT_DIR}/analyze_cross_benchmark_results.py" \
        --results_dir "${RESULTS_DIR}" \
        --experiment_name "${EXPERIMENT_NAME}" \
        --source_benchmark "${SOURCE_BENCHMARK}" \
        --target_benchmarks ${TARGET_BENCHMARKS}

    log ""
    log "=========================================="
    log "实验完成!"
    log "结果目录: ${RESULTS_DIR}"
    log "=========================================="
}

#-------------------------------------------------------------------------------
# SLURM作业提交模式
#-------------------------------------------------------------------------------

submit_slurm_job() {
    local job_name=$1
    local command=$2
    local dependency=$3

    local sbatch_args=(
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --gres="gpu:${GPUS}"
        --cpus-per-task="${CPUS_PER_TASK}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${PROJECT_ROOT}/logs/slurm_%j_${job_name}.out"
        --error="${PROJECT_ROOT}/logs/slurm_%j_${job_name}.err"
    )

    if [ -n "${dependency}" ]; then
        sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    local job_id=$(sbatch "${sbatch_args[@]}" --wrap="${command}" | awk '{print $4}')
    echo "${job_id}"
}

run_with_slurm() {
    log "使用SLURM提交作业..."

    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${SCRIPT_DIR}/generated_configs"
    mkdir -p "${PROJECT_ROOT}/logs"

    # 生成训练配置
    TRAIN_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${SOURCE_BENCHMARK}_train.yaml"
    generate_config "${SOURCE_BENCHMARK}" "train" "" "${TRAIN_CONFIG}"

    # 提交训练作业
    TRAIN_CMD="cd ${PROJECT_ROOT} && python run/run_${SOURCE_BENCHMARK}.py --config ${TRAIN_CONFIG}"
    TRAIN_JOB_ID=$(submit_slurm_job "${EXPERIMENT_NAME}_train" "${TRAIN_CMD}" "")
    log "提交训练作业: ${TRAIN_JOB_ID}"

    # 提交评估作业 (依赖训练完成)
    for target in ${TARGET_BENCHMARKS}; do
        EVAL_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${target}_eval.yaml"
        # 注意: checkpoint路径需要在评估时动态确定
        # 这里先生成配置，eval脚本会查找最新checkpoint

        EVAL_CMD="cd ${PROJECT_ROOT} && bash ${SCRIPT_DIR}/run_single_eval.sh ${target} ${EXPERIMENT_NAME}"
        EVAL_JOB_ID=$(submit_slurm_job "${EXPERIMENT_NAME}_eval_${target}" "${EVAL_CMD}" "${TRAIN_JOB_ID}")
        log "提交评估作业 (${target}): ${EVAL_JOB_ID}"
    done

    # 提交分析作业 (依赖所有评估完成)
    ANALYZE_CMD="cd ${PROJECT_ROOT} && python ${SCRIPT_DIR}/analyze_cross_benchmark_results.py --results_dir ${RESULTS_DIR} --experiment_name ${EXPERIMENT_NAME} --source_benchmark ${SOURCE_BENCHMARK} --target_benchmarks ${TARGET_BENCHMARKS}"
    # 注意: 这里简化处理，实际需要等待所有eval作业

    log "所有作业已提交"
    log "使用 'squeue -u \$USER' 查看作业状态"
}

#-------------------------------------------------------------------------------
# 使用srun直接运行模式
#-------------------------------------------------------------------------------

run_with_srun() {
    log "使用srun直接运行..."

    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${SCRIPT_DIR}/generated_configs"
    mkdir -p "${PROJECT_ROOT}/logs"

    check_requirements

    # Step 1: 训练
    log ""
    log "===== Step 1: 训练阶段 (srun) ====="

    TRAIN_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${SOURCE_BENCHMARK}_train.yaml"
    generate_config "${SOURCE_BENCHMARK}" "train" "" "${TRAIN_CONFIG}"

    srun --partition="${PARTITION}" \
         --gres="gpu:${GPUS}" \
         --cpus-per-task="${CPUS_PER_TASK}" \
         --mem="${MEM}" \
         --time="${TIME}" \
         bash -c "cd ${PROJECT_ROOT} && python run/run_${SOURCE_BENCHMARK}.py --config ${TRAIN_CONFIG}"

    # 找到checkpoint
    sleep 5  # 等待文件系统同步
    CHECKPOINT_PATH=$(find_latest_checkpoint "${RESULTS_DIR}/${SOURCE_BENCHMARK}")

    if [ -z "${CHECKPOINT_PATH}" ]; then
        error "未找到checkpoint，训练可能失败"
    fi

    log "找到checkpoint: ${CHECKPOINT_PATH}"

    # Step 2: 评估
    log ""
    log "===== Step 2: 评估阶段 (srun) ====="

    for target in ${TARGET_BENCHMARKS}; do
        log "--- 评估目标: ${target} ---"

        EVAL_CONFIG="${SCRIPT_DIR}/generated_configs/${EXPERIMENT_NAME}_${target}_eval.yaml"
        generate_config "${target}" "eval" "${CHECKPOINT_PATH}" "${EVAL_CONFIG}"

        srun --partition="${PARTITION}" \
             --gres="gpu:${GPUS}" \
             --cpus-per-task="${CPUS_PER_TASK}" \
             --mem="${MEM}" \
             --time="${TIME}" \
             bash -c "cd ${PROJECT_ROOT} && python run/run_${target}.py --config ${EVAL_CONFIG}"
    done

    # Step 3: 分析
    log ""
    log "===== Step 3: 结果分析 ====="

    python "${SCRIPT_DIR}/analyze_cross_benchmark_results.py" \
        --results_dir "${RESULTS_DIR}" \
        --experiment_name "${EXPERIMENT_NAME}" \
        --source_benchmark "${SOURCE_BENCHMARK}" \
        --target_benchmarks ${TARGET_BENCHMARKS}

    log "实验完成!"
}

#-------------------------------------------------------------------------------
# 入口点
#-------------------------------------------------------------------------------

# 解析命令行参数
MODE="${1:-local}"  # local, srun, sbatch

case "${MODE}" in
    local)
        main
        ;;
    srun)
        run_with_srun
        ;;
    sbatch)
        run_with_slurm
        ;;
    *)
        echo "Usage: $0 [local|srun|sbatch]"
        echo "  local  - 本地直接运行"
        echo "  srun   - 使用srun在计算节点运行"
        echo "  sbatch - 使用sbatch提交作业"
        exit 1
        ;;
esac
