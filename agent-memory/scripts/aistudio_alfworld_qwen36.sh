#!/bin/bash
# ============================================================================
# AIStudio: Qwen3.6 ALFWorld — 6 experiments in parallel on 8x H200
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (shared by all experiments)
#   GPU 1: Qwen3.6 vLLM → region (from scratch)
#   GPU 2: Qwen3.6 vLLM → memrl (from scratch)
#   GPU 3: Qwen3.6 vLLM → memp (from scratch)
#   GPU 4: Qwen3.6 vLLM → mem0 (from scratch)
#   GPU 5: Qwen3.6 vLLM → pass@10 (resume R1 b787)
#   GPU 6: Qwen3.6 vLLM → Self-RAG (from scratch)
#   GPU 7: Qwen3-Embedding-8B (shared, load-balance with GPU 0)
#   GPU 5: Qwen3.6 vLLM → pass@10 (from scratch)
#   GPU 6: Qwen3.6 vLLM → nomem (eval-only)
#   GPU 7: spare
#
# Eval policy: E1-E9 eval 1x temp=0; E10 eval 1x temp=0 + 3x temp=0.2
# ============================================================================
set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen36_${1:-$(date +%Y%m%d_%H%M%S)}.log"

EMBED_PORT=9000
EMBED_PORT_2=9001  # second embed on GPU 7
LLM_PORT_1=9101  # region
LLM_PORT_2=9102  # memrl
LLM_PORT_3=9103  # memp
LLM_PORT_4=9104  # mem0
LLM_PORT_5=9105  # pass@10
LLM_PORT_6=9106  # selfrag

# Fixed RUN_ID so platform retries reuse the same checkpoint directory
# (auto-resume picks up latest snapshot inside ck_dir)
export MEMRL_RUN_ID="qwen36_v9"

# --- Logging to /storage so we can tail from login node ---
mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen3.6 ALFWorld — 6 parallel experiments"
echo "Start: $(date)"
echo "=========================================="

# --- Environment setup (areal-runtime container) ---
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH=$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:$PYTHONPATH

# --- Install dependencies into venv ---
cd $MEMRL_DIR
pip install --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ . 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; print(memrl.__file__); import memos; print('[OK] memrl + memos imported')"

# --- Launch shared Embedding servers on GPU 0 and GPU 7 ---
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &
PID_EMBED=$!

CUDA_VISIBLE_DEVICES=7 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT_2 \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &
PID_EMBED2=$!

# --- Launch 6 Qwen3.6 LLM servers on GPU 1-6 ---
for gpu in 1 2 3 4 5 6; do
    port_var="LLM_PORT_${gpu}"
    port=${!port_var}
    CUDA_VISIBLE_DEVICES=$gpu python3 -m vllm.entrypoints.openai.api_server \
        --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
        --host 127.0.0.1 --port $port \
        --max-model-len 32768 --trust-remote-code \
        --reasoning-parser qwen3 &
done

# --- Wait for all servers ---
echo "[INFO] Waiting for Embedding servers..."
for eport in $EMBED_PORT $EMBED_PORT_2; do
    for i in $(seq 1 1200); do
        curl -s "http://localhost:${eport}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed port $eport ready!" && break
        [ "$i" -eq 1200 ] && echo "[ERROR] Embed port $eport timeout" && exit 1
        sleep 1
    done
done

for gpu in 1 2 3 4 5 6; do
    port_var="LLM_PORT_${gpu}"
    port=${!port_var}
    echo "[INFO] Waiting for LLM on port $port (GPU $gpu)..."
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] GPU $gpu LLM ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] GPU $gpu LLM timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All 6 vLLM servers ready."

# ============================================================================
# Helper: generate config YAML
# ============================================================================
gen_config() {
    local llm_port=$1 experiment_name=$2 mode=$3 num_sections=$4
    local k_retrieve=$5 enable_vd=$6 baseline_mode=$7 ckpt_path=$8 embed_port=${9:-$EMBED_PORT}
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
  base_url: http://localhost:${embed_port}/v1/
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
# Launch experiments in parallel
# ============================================================================

# --- 1. Region (GPU 1, resume from S5 ckpt) ---
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

# --- 2. MemRL (GPU 2, resume from S5 ckpt) ---
(
    CFG=$(gen_config $LLM_PORT_2 "alfworld_memrl_qwen36" "train" 10 3 true null "")
    echo "[EXP] MemRL starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] MemRL exit=$? at $(date)"
) &
PID_MEMRL=$!

# --- 3. MemP (GPU 3, resume from ckpt) ---
(
    CFG=$(gen_config $LLM_PORT_3 "alfworld_memp_qwen36" "train" 10 3 false null "")
    # MemP: no Q-value, pure sim retrieval, no threshold
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] MemP starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] MemP exit=$? at $(date)"
) &
PID_MEMP=$!

# --- 4. Mem0 (GPU 4, from scratch) ---
(
    CFG=$(gen_config $LLM_PORT_4 "alfworld_mem0_qwen36" "train" 10 3 false null "" $EMBED_PORT_2)
    echo "[EXP] Mem0 starting..."
    python3 run/run_alfworld.py --config "$CFG" --mem0 --skip_initial_eval
    echo "[EXP] Mem0 exit=$? at $(date)"
) &
PID_MEM0=$!

# --- 5. Pass@10 (GPU 5, from scratch) ---
(
    CFG=$(gen_config $LLM_PORT_5 "alfworld_passk10_qwen36" "train" 10 0 false "passk" "" $EMBED_PORT_2)
    sed -i 's/random_seed: 42/random_seed: 43/' "$CFG"
    sed -i 's/save_trajectories: true/save_trajectories: false/' "$CFG"
    sed -i 's/save_memories: true/save_memories: false/' "$CFG"
    echo "[EXP] Pass@10 starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] Pass@10 exit=$? at $(date)"
) &
PID_PASSK=$!

# --- 6. Self-RAG (GPU 6, from scratch) ---
(
    CFG=$(gen_config $LLM_PORT_6 "alfworld_selfrag_qwen36" "train" 10 3 false null "" $EMBED_PORT_2)
    # Self-RAG: normal section training + LLM critique filtering, no Q-value
    sed -i 's/build_strategy: proceduralization/build_strategy: trajectory/' "$CFG"
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] Self-RAG starting..."
    python3 run/run_alfworld.py --config "$CFG" --selfrag --skip_initial_eval
    echo "[EXP] Self-RAG exit=$? at $(date)"
) &
PID_SELFRAG=$!

# --- Wait for all experiments ---
echo "[INFO] All 6 experiments launched. PIDs: region=$PID_REGION memrl=$PID_MEMRL memp=$PID_MEMP mem0=$PID_MEM0 passk=$PID_PASSK selfrag=$PID_SELFRAG"

wait $PID_REGION; echo "[DONE] Region: $?"
wait $PID_MEMRL; echo "[DONE] MemRL: $?"
wait $PID_MEMP; echo "[DONE] MemP: $?"
wait $PID_MEM0; echo "[DONE] Mem0: $?"
wait $PID_PASSK; echo "[DONE] Pass@10: $?"
wait $PID_SELFRAG; echo "[DONE] Self-RAG: $?"

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
