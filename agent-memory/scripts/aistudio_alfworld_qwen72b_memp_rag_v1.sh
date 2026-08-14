#!/bin/bash
# ============================================================================
# AIStudio: Qwen2.5-72B ALFWorld MemP + RAG (vLLM, 5x H200)
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (vLLM)
#   GPU 1-2: Qwen2.5-72B TP=2 → MemP
#   GPU 3-4: Qwen2.5-72B TP=2 → RAG
# ============================================================================
set -uo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_memp_rag_supervisor_${RUN_TAG}.log"
MEMP_LOG="$MEMRL_DIR/logs/aistudio_qwen72b_memp_${RUN_TAG}.log"
RAG_LOG="$MEMRL_DIR/logs/aistudio_qwen72b_rag_${RUN_TAG}.log"

EMBED_PORT="${EMBED_PORT:-19110}"
LLM_PORT_1="${MEMP_LLM_PORT:-19310}"  # MemP
LLM_PORT_2="${RAG_LLM_PORT:-19311}"  # RAG

TS="$RUN_TAG"
MEMP_RUN_ID="${MEMP_RUN_ID:-qwen72b_memp_v1}"
RAG_RUN_ID="${RAG_RUN_ID:-qwen72b_rag_v1}"
SERVER_PIDS=()
EXP_PIDS=()
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  ((${#EXP_PIDS[@]})) && kill "${EXP_PIDS[@]}" 2>/dev/null || true
  ((${#SERVER_PIDS[@]})) && kill "${SERVER_PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "[CLEANUP] services stopped; rc=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "AIStudio: Qwen2.5-72B MemP + RAG (vLLM, 5 GPU)"
echo "Start: $(date)"
echo "Supervisor log: $LOGFILE"
echo "MemP log: $MEMP_LOG"
echo "RAG log: $RAG_LOG"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

cd $MEMRL_DIR
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

# --- Embedding (GPU 0) ---
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &
PID_EMBED=$!; SERVER_PIDS+=("$PID_EMBED")

# --- LLM servers (GPU 1-2, 3-4) ---
CUDA_VISIBLE_DEVICES=1,2 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port $LLM_PORT_1 \
    --max-model-len 32768 --trust-remote-code \
    --tensor-parallel-size 2 &
PID_LLM1=$!; SERVER_PIDS+=("$PID_LLM1")

CUDA_VISIBLE_DEVICES=3,4 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port $LLM_PORT_2 \
    --max-model-len 32768 --trust-remote-code \
    --tensor-parallel-size 2 &
PID_LLM2=$!; SERVER_PIDS+=("$PID_LLM2")

# --- Wait for servers ---
echo "[INFO] Waiting for Embed..."
for i in $(seq 1 1200); do
    kill -0 "$PID_EMBED" 2>/dev/null || { echo "[ERROR] Embed server exited before readiness"; exit 1; }
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done
for spec in "$LLM_PORT_1:$PID_LLM1" "$LLM_PORT_2:$PID_LLM2"; do
    port="${spec%%:*}"
    pid="${spec##*:}"
    echo "[INFO] Waiting for LLM on port $port..."
    for i in $(seq 1 1800); do
        kill -0 "$pid" 2>/dev/null || { echo "[ERROR] LLM server on port $port exited before readiness"; exit 1; }
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Port $port ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] Port $port timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All servers ready."

# ============================================================================
# 1. MemP (proceduralization, pure sim, no Q)
# ============================================================================
(
    export MEMRL_RUN_ID="$MEMP_RUN_ID"
    cat > /tmp/alf_memp_vllm_$$.yaml << CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT_1}/v1/
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
  build_strategy: proceduralization
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
  enable_value_driven: false
  experiment_name: alfworld_memp_qwen72b
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
  tau: 0.0
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
  weight_sim: 1.0
  weight_q: 0.0
CFGEOF
    echo "[EXP] 72B MemP starting at $(date)" > "$MEMP_LOG"
    python3 run/run_alfworld.py --config /tmp/alf_memp_vllm_$$.yaml --skip_initial_eval >> "$MEMP_LOG" 2>&1
    rc=$?; echo "[EXP] 72B MemP exit=$rc at $(date)" >> "$MEMP_LOG"; exit "$rc"
) &
PID_MEMP=$!; EXP_PIDS+=("$PID_MEMP")

# ============================================================================
# 2. RAG (trajectory build, pure sim, no Q)
# ============================================================================
(
    export MEMRL_RUN_ID="$RAG_RUN_ID"
    cat > /tmp/alf_rag_vllm_$$.yaml << CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT_2}/v1/
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
  enable_value_driven: false
  experiment_name: alfworld_rag_qwen72b
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
  tau: 0.0
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
  weight_sim: 1.0
  weight_q: 0.0
CFGEOF
    echo "[EXP] 72B RAG starting at $(date)" > "$RAG_LOG"
    python3 run/run_alfworld.py --config /tmp/alf_rag_vllm_$$.yaml --skip_initial_eval >> "$RAG_LOG" 2>&1
    rc=$?; echo "[EXP] 72B RAG exit=$rc at $(date)" >> "$RAG_LOG"; exit "$rc"
) &
PID_RAG=$!; EXP_PIDS+=("$PID_RAG")

echo "[INFO] 2 experiments launched. PIDs: memp=$PID_MEMP rag=$PID_RAG"

wait "$PID_MEMP"; RC_MEMP=$?; echo "[DONE] MemP: $RC_MEMP"
wait "$PID_RAG"; RC_RAG=$?; echo "[DONE] RAG: $RC_RAG"
EXP_PIDS=()
if ((RC_MEMP != 0 || RC_RAG != 0)); then
  echo "[ERROR] combined result MemP=$RC_MEMP RAG=$RC_RAG"
  exit 1
fi

echo "=========================================="
echo "All done. End: $(date)"
echo "=========================================="
