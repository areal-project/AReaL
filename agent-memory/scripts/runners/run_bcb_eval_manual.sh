#!/bin/bash
# Manually run BCB evaluation on HLE and ALF
# Usage: bash run_bcb_eval_manual.sh

set -ex

cd /storage/openpsi/users/yl/agent-memory/MemRL

# BCB→HLE checkpoint (490 memories, trained on BCB)
BCB_HLE_CHECKPOINT="/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000905_gpt-4o-2024-11-20_rl-on/snapshot/final"

# BCB→ALF checkpoint (509 memories, trained on BCB)
BCB_ALF_CHECKPOINT="/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_hard/memory/20260420_000926_gpt-4o-2024-11-20_rl-on/snapshot/final"

# Install dependencies
pip install -e . -q 2>/dev/null
pip install -r requirements.txt -q 2>/dev/null

# Install and start LiteLLM
pip install 'litellm[proxy]' -q 2>/dev/null

LITELLM_PORT=$((10000 + RANDOM % 50000))
TEMP_CONFIG="/tmp/litellm_config_bcb_eval.yaml"
cp /storage/openpsi/users/yl/.claude/config.yaml ${TEMP_CONFIG}

echo "[INFO] Starting LiteLLM on port ${LITELLM_PORT}..."
python -m litellm.proxy.proxy_cli --config ${TEMP_CONFIG} --port ${LITELLM_PORT} --host 127.0.0.1 > /tmp/litellm_bcb_eval.log 2>&1 &
LITELLM_PID=$!

# Wait for LiteLLM
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}"
for i in $(seq 1 60); do
    if curl -s ${LITELLM_URL}/health > /dev/null 2>&1; then
        echo "[INFO] LiteLLM ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[ERROR] LiteLLM timeout"
        kill ${LITELLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 2
done

EVAL_TARGET=$1

if [ "$EVAL_TARGET" == "hle" ] || [ -z "$EVAL_TARGET" ]; then
    echo "[INFO] Running BCB→HLE evaluation with checkpoint: ${BCB_HLE_CHECKPOINT}"

    # Generate HLE eval config
    cat > /tmp/bcb_hle_eval_config.yaml << EOF
llm:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'gpt-4o-2024-11-20'
  temperature: 0.0
  max_tokens: 2048

embedding:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'text-embedding-3-small'

memory:
  load_from: '${BCB_HLE_CHECKPOINT}'
  sim_norm_params:
    mean: 0.19
    std: 0.09

hle:
  parquet_path: 'data/hle/hle_test.parquet'
  dataset_ratio: 1.0
  mode: 'test'

experiment:
  exp_name: 'bcb_to_hle_eval'
  num_sections: 1
  batch_size: 5
  random_seed: 42
  retrieve_k: 10
  rl_enabled: false
EOF

    python run/run_hle.py --config /tmp/bcb_hle_eval_config.yaml
fi

if [ "$EVAL_TARGET" == "alf" ] || [ -z "$EVAL_TARGET" ]; then
    echo "[INFO] Running BCB→ALF evaluation with checkpoint: ${BCB_ALF_CHECKPOINT}"

    # Generate ALF eval config
    cat > /tmp/bcb_alf_eval_config.yaml << EOF
llm:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'gpt-4o-2024-11-20'
  temperature: 0.0
  max_tokens: 2048

embedding:
  api_key: 'sk-placeholder'
  base_url: '${LITELLM_URL}'
  model: 'text-embedding-3-small'

memory:
  load_from: '${BCB_ALF_CHECKPOINT}'
  sim_norm_params:
    mean: 0.52
    std: 0.12

alf:
  num_envs: 3
  max_steps: 50
  data_path: 'data/alfworld/json_2.1.1/valid_seen'
  mode: 'test'
  react_style: true

experiment:
  exp_name: 'bcb_to_alf_eval'
  num_sections: 1
  batch_size: 5
  random_seed: 42
  retrieve_k: 10
  rl_enabled: false
EOF

    python run/run_alfworld.py --config /tmp/bcb_alf_eval_config.yaml
fi

# Cleanup
echo "[INFO] Stopping LiteLLM..."
kill ${LITELLM_PID} 2>/dev/null
rm -f ${TEMP_CONFIG}

echo "[INFO] BCB evaluation completed"
