#!/bin/bash
# Resume ALFWorld Qwen2.5-72B trajectory MemRL from the latest batch checkpoint
# of run 20260716-153758 (currently s9_b20), and finish S9-S10.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_memrl_traj_resume_s9_${TS}.log"
EMBED_PORT="${EMBED_PORT:-9000}"
LLM_PORT="${LLM_PORT:-9201}"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Qwen2.5-72B trajectory MemRL resume S9-S10"
echo "Run ID: 20260716-153758"
echo "Start: $(date)"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
USER_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:$VENV_SP:$USER_SP:${PYTHONPATH:-}"
export MEMRL_RUN_ID="20260716-153758"

cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 - <<'PYCODE'
import memos, memrl, vllm
print("[PREFLIGHT] memrl:", memrl.__file__)
print("[PREFLIGHT] memos:", memos.__file__)
print("[PREFLIGHT] vllm:", vllm.__file__)
PYCODE

SERVER_PIDS=()
cleanup() {
    status=$?
    echo "[INFO] cleanup status=$status at $(date)"
    if ((${#SERVER_PIDS[@]})); then kill "${SERVER_PIDS[@]}" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" --max-model-len 8192 \
    --trust-remote-code --convert embed &
PID_EMBED=$!; SERVER_PIDS+=("$PID_EMBED")

CUDA_VISIBLE_DEVICES=1,2 python3 -m vllm.entrypoints.openai.api_server \
    --model "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port "$LLM_PORT" --max-model-len 32768 \
    --trust-remote-code --tensor-parallel-size 2 &
PID_LLM=$!; SERVER_PIDS+=("$PID_LLM")

wait_for_server() {
    local name=$1 port=$2 pid=$3 expected_model=$4 timeout=$5
    local response stable=0
    echo "[WAIT] $name pid=$pid port=$port model=$expected_model"
    for ((i=1; i<=timeout; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[FATAL] $name process $pid exited before readiness"
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        response=$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
        if MODEL_RESPONSE="$response" EXPECTED_MODEL="$expected_model" python3 - <<'PYREADY'
import json, os, sys
try:
    ids = {x.get("id") for x in json.loads(os.environ["MODEL_RESPONSE"]).get("data", [])}
except Exception:
    sys.exit(1)
sys.exit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)
PYREADY
        then
            stable=$((stable + 1))
            if ((stable >= 5)); then
                echo "[READY] $name is owned by pid=$pid"
                return 0
            fi
        else
            stable=0
        fi
        sleep 1
    done
    echo "[FATAL] timeout waiting for $name"
    return 1
}

wait_for_server embed "$EMBED_PORT" "$PID_EMBED" Qwen/Qwen3-Embedding-8B 1200
wait_for_server llm "$LLM_PORT" "$PID_LLM" Qwen2.5-72B-Instruct 1800


CFG="/tmp/alf_memrl_traj_resume_s9_$$.yaml"
cat > "$CFG" <<CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT}/v1/
  model: Qwen2.5-72B-Instruct
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${EMBED_PORT}/v1/
  model: Qwen/Qwen3-Embedding-8B
  max_text_len: 8196
  dimension: 4096
memory:
  build_strategy: trajectory
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 3
  max_keywords: 5
  add_similarity_threshold: 0.9
  memory_budget_tokens: 0
  sim_norm_mean: 0.5187
  sim_norm_std: 0.1203
environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv
experiment:
  random_seed: 42
  enable_value_driven: true
  experiment_name: alfworld_memrl_traj_qwen72b
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config:
  epsilon: 0
  tau: 0.62
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0
  q_init_neg: 0
  success_reward: 1.0
  failure_reward: -1.0
  topk: 3
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -10
  weight_sim: 0.5
  weight_q: 0.5
CFGEOF

python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
rc=$?
echo "[EXP] trajectory MemRL resume exit=${rc} at $(date)"
exit "$rc"
