#!/bin/bash
# ============================================================================
# AIStudio: Qwen3.6 ALFWorld — 4 experiments on 5x GPU (Pass@10 completed)
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (shared)
#   GPU 1: Qwen3.6 vLLM (reasoning-parser) → Region+FS
#   GPU 2: Qwen3.6 vLLM (reasoning-parser) → MemRL
#   GPU 3: Qwen3.6 vLLM (reasoning-parser) → MemP
#   GPU 4: Qwen3.6 vLLM (reasoning-parser) → RAG
# ============================================================================
set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen36_main_${1:-$(date +%Y%m%d_%H%M%S)}.log"

EMBED_PORT=9000
LLM_PORT_1=9101  # region
LLM_PORT_2=9102  # memrl
LLM_PORT_3=9103  # memp
LLM_PORT_4=9104  # rag

# Resume from v9 checkpoints
export MEMRL_RUN_ID="qwen36_v9"

# --- Logging ---
mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen3.6 ALFWorld — 4 experiments (Region, MemRL, MemP, RAG)"
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

# --- Launch Embedding server on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &

# --- Launch 4 Qwen3.6 LLM servers on GPU 1-4 (all with reasoning-parser) ---
for gpu in 1 2 3 4; do
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

for gpu in 1 2 3 4; do
    port_var="LLM_PORT_${gpu}"
    port=${!port_var}
    echo "[INFO] Waiting for LLM on port $port (GPU $gpu)..."
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] GPU $gpu LLM ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] GPU $gpu LLM timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All 4 vLLM servers ready."

# ============================================================================
# Helper: generate config YAML
# ============================================================================
gen_config() {
    local llm_port=$1 experiment_name=$2 mode=$3 num_sections=$4
    local k_retrieve=$5 enable_vd=$6 baseline_mode=$7 ckpt_path=$8
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
  max_recent_turns: 20
  strip_thinking: true
  max_trajectory_len: 6000
  max_history_response_chars: 4000
  force_think: false
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: $([ -n "$ckpt_path" ] && echo "true" || echo "false")
  ckpt_resume_path: "${ckpt_path}"
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
# Launch 5 experiments (resume from v9 checkpoints)
# ============================================================================

# --- 1. Region+FS (GPU 1) ---
(
    CFG=$(gen_config $LLM_PORT_1 "alfworld_region_qwen36" "train" 10 3 true null "")
    echo "[EXP] Region starting..."
    python3 run/run_alfworld.py \
        --config "$CFG" \
        --region --region_gating_mode additive \
        --region_utility_mode beta \
        --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
        --val_lambda_max 0.05 --no_z_norm \
        --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
        --failure_summary_n_slots 2 \
        --skip_initial_eval
    echo "[EXP] Region exit=$? at $(date)"
) &
PID_REGION=$!

# --- 2. MemRL (GPU 2) ---
(
    CFG=$(gen_config $LLM_PORT_2 "alfworld_memrl_qwen36" "train" 10 3 true null "")
    echo "[EXP] MemRL starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] MemRL exit=$? at $(date)"
) &
PID_MEMRL=$!

# --- 3. MemP (GPU 3) ---
(
    CFG=$(gen_config $LLM_PORT_3 "alfworld_memp_qwen36" "train" 10 3 false null "")
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] MemP starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] MemP exit=$? at $(date)"
) &
PID_MEMP=$!

# --- 4. RAG (GPU 4) ---
(
    CFG=$(gen_config $LLM_PORT_4 "alfworld_rag_pure_qwen36" "train" 10 3 false null "")
    sed -i 's/build_strategy: proceduralization/build_strategy: trajectory/' "$CFG"
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] RAG starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] RAG exit=$? at $(date)"
) &
PID_RAG=$!

# --- Wait ---
echo "[INFO] 4 experiments launched. PIDs: region=$PID_REGION memrl=$PID_MEMRL memp=$PID_MEMP rag=$PID_RAG"

wait $PID_REGION; echo "[DONE] Region: $?"
wait $PID_MEMRL; echo "[DONE] MemRL: $?"
wait $PID_MEMP; echo "[DONE] MemP: $?"
wait $PID_RAG; echo "[DONE] RAG: $?"

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
