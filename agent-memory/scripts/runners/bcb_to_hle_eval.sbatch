#!/bin/bash
#SBATCH --job-name=bcb_hle_eval
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/bcb_hle_eval_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/bcb_hle_eval_%j.err

set -ex

echo "[INFO] Starting BCB→HLE evaluation on $(hostname) at $(date)"
cd /storage/openpsi/users/yl/agent-memory/MemRL

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash << 'INNER_SCRIPT'
set -ex
cd /storage/openpsi/users/yl/agent-memory/MemRL

# Install dependencies
pip install -e . -q 2>/dev/null || true
pip install -r requirements.txt -q 2>/dev/null || true
pip install 'litellm[proxy]' -q 2>/dev/null || true

# Start LiteLLM
LITELLM_PORT=$((10000 + RANDOM % 50000))
TEMP_CONFIG="/tmp/litellm_config_bcb_hle.yaml"
cp /storage/openpsi/users/yl/.claude/config.yaml ${TEMP_CONFIG}

echo "[INFO] Starting LiteLLM on port ${LITELLM_PORT}..."
python -m litellm.proxy.proxy_cli --config ${TEMP_CONFIG} --port ${LITELLM_PORT} --host 127.0.0.1 > /tmp/litellm_bcb_hle.log 2>&1 &
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
        cat /tmp/litellm_bcb_hle.log
        kill ${LITELLM_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Generate HLE eval config with correct format
cat > /tmp/bcb_hle_eval_config.yaml << EOF
llm:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'gpt-4o-2024-11-20'
  provider: 'openai'
  temperature: 0.0
  max_tokens: 10240

embedding:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'text-embedding-3-small'
  provider: 'openai'
  max_text_len: 8196

memory:
  checkpoint_path: '/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000905_gpt-4o-2024-11-20_rl-on/snapshot/final'
  load_from_checkpoint: true
  k_retrieve: 10
  user_id: 'bcb_to_hle_eval'
  sim_norm_mean: 0.19
  sim_norm_std: 0.09
  retrieve_strategy: 'query'
  build_strategy: 'proceduralization'
  update_strategy: 'adjustment'

experiment:
  experiment_name: 'bcb_to_hle_eval'
  mode: 'test'
  num_sections: 1
  batch_size: 5
  random_seed: 42
  dataset_ratio: 1.0
  split_file: 'data/hle/hle_test.parquet'
  output_dir: '/storage/openpsi/users/yl/agent-memory/MemRL/results'
  enable_value_driven: false
  save_memories: true
  save_trajectories: true
  ckpt_resume_enabled: true
  ckpt_resume_path: '/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000905_gpt-4o-2024-11-20_rl-on/snapshot/final'

rl_config:
  alpha: 0.3
  gamma: 0.0
  tau: 0.35
  epsilon: 0.0
  success_reward: 1.0
  failure_reward: 0.0
  weight_q: 0.5
  weight_sim: 0.5
  topk: 5
EOF

echo "[INFO] Running HLE evaluation..."
python run/run_hle.py --config /tmp/bcb_hle_eval_config.yaml

# Cleanup
echo "[INFO] Stopping LiteLLM..."
kill ${LITELLM_PID} 2>/dev/null || true

echo "[INFO] BCB→HLE evaluation completed at $(date)"
INNER_SCRIPT
