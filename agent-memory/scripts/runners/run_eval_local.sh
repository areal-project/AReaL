#!/bin/bash
# 本地运行评估脚本（不需要GPU，直接在登录节点运行）
# 使用 Singularity 容器确保环境一致
# 用法: ./run_eval_local.sh <eval_type> [checkpoint_path]
# eval_type: alf 或 hle
# checkpoint_path: 可选，默认使用 BCB 训练的 checkpoint

set -e

EVAL_TYPE=${1:-"alf"}
CHECKPOINT_PATH=${2:-"/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000905_gpt-4o-2024-11-20_rl-on/snapshot/final"}

# 使用 Singularity 容器运行
CONTAINER="/storage/openpsi/images/areal-latest.sif"

if [ ! -f "${CONTAINER}" ]; then
    echo "[ERROR] Container not found: ${CONTAINER}"
    exit 1
fi

echo "[INFO] Running ${EVAL_TYPE} evaluation in Singularity container..."
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"

singularity exec --bind /storage:/storage ${CONTAINER} \
    bash -c "
set -e
cd /storage/openpsi/users/yl/agent-memory/MemRL

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

# 安装依赖
pip install -e . -q 2>&1 | tail -3
pip install 'litellm[proxy]' -q 2>&1 | tail -3

# 检查 LiteLLM 配置
LITELLM_CONFIG='/storage/openpsi/users/yl/.claude/config.yaml'
if [ ! -f \"\${LITELLM_CONFIG}\" ]; then
    echo '[ERROR] LiteLLM config not found'
    exit 1
fi

# 启动 LiteLLM 服务
LITELLM_PORT=\$((10000 + RANDOM % 50000))
LITELLM_URL=\"http://127.0.0.1:\${LITELLM_PORT}\"
LOG_FILE=\"/tmp/litellm_eval_${EVAL_TYPE}_\$\$.log\"

echo '[INFO] Starting LiteLLM on port '\${LITELLM_PORT}'...'
python3 -m litellm.proxy.proxy_cli --config \${LITELLM_CONFIG} --port \${LITELLM_PORT} --host 127.0.0.1 > \${LOG_FILE} 2>&1 &
LITELLM_PID=\$!

# 等待 LiteLLM 就绪
echo '[INFO] Waiting for LiteLLM...'
for i in \$(seq 1 60); do
    if curl -s \${LITELLM_URL}/health > /dev/null 2>&1; then
        echo '[INFO] LiteLLM ready at '\${LITELLM_URL}
        break
    fi
    if [ \$i -eq 60 ]; then
        echo '[ERROR] LiteLLM timeout'
        cat \${LOG_FILE}
        kill \${LITELLM_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# 清理函数
cleanup() {
    echo '[INFO] Stopping LiteLLM...'
    kill \${LITELLM_PID} 2>/dev/null || true
}
trap cleanup EXIT

# 生成配置文件
CONFIG_FILE=\"/tmp/eval_${EVAL_TYPE}_config_\$\$.yaml\"

if [ '${EVAL_TYPE}' == 'alf' ]; then
    cat > \${CONFIG_FILE} << EOF
llm:
  provider: openai
  api_key: sk-placeholder
  base_url: \${LITELLM_URL}
  model: gpt-4o-2024-11-20
  temperature: 0.0
  max_tokens: 10240

embedding:
  provider: openai
  api_key: sk-placeholder
  base_url: \${LITELLM_URL}
  model: text-embedding-3-small
  max_text_len: 6000

memory:
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 10
  max_keywords: 8
  add_similarity_threshold: 0.99
  load_from_checkpoint: true
  checkpoint_path: ${CHECKPOINT_PATH}
  sim_norm_mean: 0.52
  sim_norm_std: 0.12

environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv

experiment:
  experiment_name: bcb_to_alf_eval_local
  algorithm: rl
  enable_value_driven: true
  random_seed: 42
  mode: test
  num_sections: 1
  batch_size: 5
  max_steps: 50
  valid_interval: 0
  test_interval: 0
  dataset_ratio: 1.0
  output_dir: /storage/openpsi/users/yl/agent-memory/MemRL/results
  few_shot_path: data/alfworld/alfworld_examples.json

rl_config:
  epsilon: 0.0
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
    echo '[INFO] Running ALFWorld evaluation...'
    python3 -B run/run_alfworld.py --config \${CONFIG_FILE}

elif [ '${EVAL_TYPE}' == 'hle' ]; then
    cat > \${CONFIG_FILE} << EOF
llm:
  provider: openai
  api_key: sk-placeholder
  base_url: \${LITELLM_URL}
  model: gpt-4o-2024-11-20
  temperature: 0.0
  max_tokens: 10240

embedding:
  provider: openai
  api_key: sk-placeholder
  base_url: \${LITELLM_URL}
  model: text-embedding-3-small
  max_text_len: 6000

memory:
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 10
  max_keywords: 8
  add_similarity_threshold: 0.99
  load_from_checkpoint: true
  checkpoint_path: ${CHECKPOINT_PATH}
  sim_norm_mean: 0.19
  sim_norm_std: 0.09

experiment:
  experiment_name: bcb_to_hle_eval_local
  algorithm: rl
  enable_value_driven: true
  random_seed: 42
  mode: test
  split_file: data/hle/hle_test.parquet
  num_sections: 1
  batch_size: 8
  max_steps: 15
  dataset_ratio: 1.0
  output_dir: /storage/openpsi/users/yl/agent-memory/MemRL/results
  train_valid_split: 0.0

rl_config:
  epsilon: 0.0
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
    echo '[INFO] Running HLE evaluation...'
    python3 -B run/run_hle.py --config \${CONFIG_FILE}

else
    echo '[ERROR] Unknown eval type: ${EVAL_TYPE}'
    echo 'Usage: run_eval_local.sh <alf|hle> [checkpoint_path]'
    exit 1
fi

echo '[INFO] Evaluation completed'
"
