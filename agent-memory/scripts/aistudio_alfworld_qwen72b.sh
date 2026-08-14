#!/bin/bash
# ============================================================================
# AIStudio: Qwen2.5-72B ALFWorld + Qwen3.6 RAG on 8x H200
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (shared)
#   GPU 1-2: Qwen2.5-72B vLLM TP=2 → region (from scratch)
#   GPU 3-4: Qwen2.5-72B vLLM TP=2 → memrl (from scratch)
#   GPU 5-6: Qwen2.5-72B vLLM TP=2 → pass@10 (from scratch)
#   GPU 7: Qwen3.6-35B-A3B → RAG (from scratch)
#
# Eval policy: E1-E9 eval 1x temp=0; E10 eval 1x temp=0 + 3x temp=0.2
# ============================================================================
set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen72b_${1:-$(date +%Y%m%d_%H%M%S)}.log"

# Stable run id: reuse submit-time $1 so platform retries write to the same
# checkpoint directory instead of creating a new timestamped exp dir.
# Convert underscore to dash to match run_alfworld.py's strftime format.
export MEMRL_RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
MEMRL_RUN_ID="${MEMRL_RUN_ID//_/-}"

EMBED_PORT=9000
LLM_PORT_1=9201  # region (72B)
LLM_PORT_2=9202  # memrl (72B)
LLM_PORT_3=9203  # pass@10 (72B)
LLM_PORT_4=9204  # RAG (Qwen3.6)

# --- Logging ---
mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen2.5-72B + Qwen3.6 RAG ALFWorld — 4 parallel experiments"
echo "Start: $(date)"
echo "=========================================="

# --- Environment setup ---
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH=$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:$PYTHONPATH

# --- Install dependencies ---
cd $MEMRL_DIR
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

# --- Launch shared Embedding server on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --context-length 8192 --trust-remote-code \
    --is-embedding &

# --- Launch 3 Qwen2.5-72B LLM servers (TP=2 each, different nccl ports) ---
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port $LLM_PORT_1 \
    --trust-remote-code --context-length 32768 \
    --nccl-port 29501 &

CUDA_VISIBLE_DEVICES=3,4 python3 -m sglang.launch_server \
    --model-path $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port $LLM_PORT_2 \
    --trust-remote-code --context-length 32768 \
    --nccl-port 29502 &

CUDA_VISIBLE_DEVICES=5,6 python3 -m sglang.launch_server \
    --model-path $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port $LLM_PORT_3 \
    --trust-remote-code --context-length 32768 \
    --nccl-port 29503 &

# --- Launch Qwen3.6 on GPU 7 for RAG baseline ---
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tp 1 --host 127.0.0.1 --port $LLM_PORT_4 \
    --trust-remote-code --context-length 65536 \
    --reasoning-parser qwen3 &

# --- Wait for all servers ---
echo "[INFO] Waiting for Embed (port $EMBED_PORT)..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done

for port in $LLM_PORT_1 $LLM_PORT_2 $LLM_PORT_3 $LLM_PORT_4; do
    echo "[INFO] Waiting for LLM on port $port..."
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Port $port ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] Port $port timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All 5 sglang servers ready."

# ============================================================================
# Helper: generate config
# ============================================================================
gen_config() {
    local llm_port=$1 experiment_name=$2 mode=$3 num_sections=$4
    local k_retrieve=$5 enable_vd=$6 baseline_mode=$7
    local cfg_path="/tmp/alf_${experiment_name}_$$.yaml"
    cat > "$cfg_path" << CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${llm_port}/v1/
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
  k_retrieve: ${k_retrieve}
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
  enable_value_driven: ${enable_vd}
  experiment_name: ${experiment_name}
  mode: ${mode}
  num_sections: ${num_sections}
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: ${baseline_mode}
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
    echo "$cfg_path"
}

# ============================================================================
# Launch experiments
# ============================================================================

# --- 1. Region (GPU 1-2, from scratch) ---
(
    CFG=$(gen_config $LLM_PORT_1 "alfworld_region_qwen72b" "train" 10 3 true null)
    echo "[EXP] 72B Region starting..."
    python3 run/run_alfworld.py \
        --config "$CFG" \
        --region --region_gating_mode additive \
        --region_utility_mode beta \
        --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
        --val_lambda_max 0.05 --no_z_norm \
        --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
        --failure_summary_n_slots 2 \
        --skip_initial_eval
    echo "[EXP] 72B Region exit=$? at $(date)"
) &
PID_REGION=$!

# --- 2. MemRL (GPU 3-4, from scratch) ---
(
    CFG=$(gen_config $LLM_PORT_2 "alfworld_memrl_qwen72b" "train" 10 3 true null)
    echo "[EXP] 72B MemRL starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B MemRL exit=$? at $(date)"
) &
PID_MEMRL=$!

# --- 3. Pass@10 (GPU 5-6, from scratch — pass@1 = no-mem SR) ---
(
    CFG=$(gen_config $LLM_PORT_3 "alfworld_passk10_qwen72b" "train" 10 0 false "passk")
    sed -i 's/random_seed: 42/random_seed: 43/' "$CFG"
    sed -i 's/save_trajectories: true/save_trajectories: false/' "$CFG"
    sed -i 's/save_memories: true/save_memories: false/' "$CFG"
    echo "[EXP] 72B Pass@10 starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B Pass@10 exit=$? at $(date)"
) &
PID_PASSK=$!

echo "[INFO] 3 experiments launched. PIDs: region=$PID_REGION memrl=$PID_MEMRL passk=$PID_PASSK"

# --- 4. Qwen3.6 RAG (GPU 7, from scratch) ---
(
    CFG_RAG="/tmp/alf_qwen36_rag_$$.yaml"
    cat > "$CFG_RAG" << CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT_4}/v1/
  model: Qwen3.6-35B-A3B
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
  experiment_name: alfworld_rag_qwen36_v2
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
    # RAG uses its own fresh run_id (from scratch, not resuming old broken run)
    unset MEMRL_RUN_ID
    echo "[EXP] Qwen3.6 RAG starting..."
    python3 run/run_alfworld.py --config "$CFG_RAG" --skip_initial_eval
    echo "[EXP] Qwen3.6 RAG exit=$? at $(date)"
) &
PID_RAG=$!

echo "[INFO] + RAG launched (PID=$PID_RAG)"

wait $PID_REGION; echo "[DONE] Region: $?"
wait $PID_MEMRL; echo "[DONE] MemRL: $?"
wait $PID_PASSK; echo "[DONE] Pass@10: $?"
wait $PID_RAG; echo "[DONE] Qwen3.6 RAG: $?"

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
