#!/bin/bash
#SBATCH --job-name=bcb_alf_eval_v3
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/bcb_alf_eval_v3_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/bcb_alf_eval_v3_%j.err

set -ex

echo "[INFO] Starting BCB→ALF evaluation (TEST MODE) on $(hostname)"
cd /storage/openpsi/users/yl/agent-memory/MemRL

singularity exec --bind /storage:/storage /storage/openpsi/images/areal-latest.sif \
    bash << 'INNER_SCRIPT'
set -ex
cd /storage/openpsi/users/yl/agent-memory/MemRL

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

# Install dependencies
# Note: Don't use 'pip install -e .' as it installs incompatible chonkie version
# Clear user site-packages to avoid version conflicts, then reinstall needed packages
rm -rf ~/.local/lib/python3.12/site-packages/chonkie* 2>/dev/null || true
export PYTHONPATH="/storage/openpsi/users/yl/agent-memory/MemRL:\${PYTHONPATH}"
pip install 'litellm[proxy]' textworld tensorboard alfworld 'chonkie<1.0' qdrant-client -q 2>&1 | tail -3

# Start LiteLLM
LITELLM_PORT=$((10000 + RANDOM % 50000))
cp /storage/openpsi/users/yl/.claude/config.yaml /tmp/litellm_alf_eval.yaml

echo "[INFO] Starting LiteLLM on port ${LITELLM_PORT}..."
python3 -m litellm.proxy.proxy_cli --config /tmp/litellm_alf_eval.yaml --port ${LITELLM_PORT} --host 127.0.0.1 > /tmp/litellm_alf_eval.log 2>&1 &
LITELLM_PID=$!

# Wait for LiteLLM
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}"
for i in $(seq 1 60); do
    if curl -s ${LITELLM_URL}/health > /dev/null 2>&1; then
        echo "[INFO] LiteLLM ready at ${LITELLM_URL}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[ERROR] LiteLLM timeout"
        cat /tmp/litellm_alf_eval.log
        exit 1
    fi
    sleep 2
done

# Generate config file with TEST mode
CONFIG_FILE="/tmp/bcb_alf_eval_config.yaml"
cat > ${CONFIG_FILE} << EOF
llm:
  provider: openai
  api_key: sk-placeholder
  base_url: ${LITELLM_URL}
  model: gpt-4o-2024-11-20
  temperature: 0.0
  max_tokens: 10240

embedding:
  provider: openai
  api_key: sk-placeholder
  base_url: ${LITELLM_URL}
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
  checkpoint_path: /storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000905_gpt-4o-2024-11-20_rl-on/snapshot/final
  sim_norm_mean: 0.52
  sim_norm_std: 0.12

environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv

experiment:
  experiment_name: bcb_to_alf_eval_v3
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

echo "[INFO] Starting ALFWorld evaluation with BCB memory (TEST MODE)..."
python3 -B run/run_alfworld.py --config ${CONFIG_FILE}

echo "[INFO] Stopping LiteLLM..."
kill ${LITELLM_PID} 2>/dev/null || true

echo "[INFO] BCB→ALF evaluation completed"
INNER_SCRIPT
