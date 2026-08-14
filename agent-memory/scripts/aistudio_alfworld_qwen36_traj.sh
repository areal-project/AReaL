#!/bin/bash
# ============================================================================
# AIStudio: Qwen3.6 ALFWorld — MemRL-traj + Region-traj (trajectory storage)
#
# Same as MemRL/Region+FS but with build_strategy=trajectory instead of
# proceduralization. Tests whether trajectory storage explains the gap
# between RAG (54%) and MemRL (41%).
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (shared)
#   GPU 1: Qwen3.6 vLLM → Region+FS (trajectory)
#   GPU 2: Qwen3.6 vLLM → MemRL (trajectory)
# ============================================================================
set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen36_traj_${1:-$(date +%Y%m%d_%H%M%S)}.log"

EMBED_PORT=9000
LLM_PORT_1=9101  # region-traj
LLM_PORT_2=9102  # memrl-traj

export MEMRL_RUN_ID="qwen36_traj_v1"

# --- Logging ---
mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen3.6 ALFWorld — MemRL-traj + Region-traj"
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
pip install --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ . 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; print(memrl.__file__); import memos; print('[OK] memrl + memos imported')"

# --- Launch Embedding server (GPU 0) ---
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &

# --- Launch 2 LLM servers (GPU 1-2) ---
for gpu in 1 2; do
    port_var="LLM_PORT_${gpu}"
    port=${!port_var}
    CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
        --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
        --host 127.0.0.1 --port $port \
        --max-model-len 32768 --trust-remote-code \
        --reasoning-parser qwen3 &
done

# --- Wait for servers ---
echo "[INFO] Waiting for Embedding server..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done

for gpu in 1 2; do
    port_var="LLM_PORT_${gpu}"
    port=${!port_var}
    echo "[INFO] Waiting for LLM on port $port..."
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Port $port ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] Port $port timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All servers ready."

# ============================================================================
# Helper
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
  max_recent_turns: 20
  strip_thinking: true
  max_trajectory_len: 6000
  max_history_response_chars: 4000
  force_think: false
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

# --- 1. Region+FS with trajectory storage (GPU 1) ---
(
    CFG=$(gen_config $LLM_PORT_1 "alfworld_region_traj_qwen36" "train" 10 3 true null)
    echo "[EXP] Region-traj starting..."
    python3 run/run_alfworld.py \
        --config "$CFG" \
        --region --region_gating_mode additive \
        --region_utility_mode beta \
        --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
        --val_lambda_max 0.05 --no_z_norm \
        --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
        --failure_summary_n_slots 2 \
        --skip_initial_eval
    echo "[EXP] Region-traj exit=$? at $(date)"
) &
PID_REGION=$!

# --- 2. MemRL with trajectory storage (GPU 2) ---
(
    CFG=$(gen_config $LLM_PORT_2 "alfworld_memrl_traj_qwen36" "train" 10 3 true null)
    echo "[EXP] MemRL-traj starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] MemRL-traj exit=$? at $(date)"
) &
PID_MEMRL=$!

# --- Wait ---
echo "[INFO] 2 experiments launched. PIDs: region=$PID_REGION memrl=$PID_MEMRL"

wait $PID_REGION; echo "[DONE] Region-traj: $?"
wait $PID_MEMRL; echo "[DONE] MemRL-traj: $?"

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
