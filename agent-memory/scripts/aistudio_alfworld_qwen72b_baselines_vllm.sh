#!/bin/bash
# ============================================================================
# AIStudio: Qwen2.5-72B ALFWorld — 5 baselines in one job (vLLM, 8x H200)
#
# Experiments:
#   1. MemRL (proc) resume — from S9 batch 20
#   2. MemRL traj — from scratch
#   3. RAG (72B) — from scratch
#   4. MemP (72B) — from scratch
#   5. Mem0 (72B) — from scratch (shared LLM for extraction)
#
# GPU layout (8 GPUs):
#   GPU 0: Qwen3-Embedding-8B (shared, vLLM)
#   GPU 1-2: Qwen2.5-72B TP=2 → MemRL proc resume
#   GPU 3-4: Qwen2.5-72B TP=2 → MemRL traj
#   GPU 5-6: Qwen2.5-72B TP=2 → RAG + MemP (sequential)
#   GPU 7: spare / Mem0 extraction if needed
#
# Note: RAG and MemP share GPU 5-6 sequentially (MemP after RAG finishes)
#       Mem0 uses the same LLM as MemRL traj (port 9202) for extraction
# ============================================================================
set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen72b_baselines_vllm_${1:-$(date +%Y%m%d_%H%M%S)}.log"

EMBED_PORT=9000
LLM_PORT_1=9201  # MemRL proc resume
LLM_PORT_2=9202  # MemRL traj
LLM_PORT_3=9203  # RAG / MemP / Mem0

TS="${1:-$(date +%Y%m%d_%H%M%S)}"

# MemRL proc resume needs the original run ID
export MEMRL_PROC_RUN_ID="20260709-151359"

mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen2.5-72B 5 baselines (vLLM, 8 GPU)"
echo "Start: $(date)"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH=$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:$PYTHONPATH

cd $MEMRL_DIR
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

# --- Launch Embedding server (GPU 0, vLLM) ---
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &

# --- Launch 3 Qwen2.5-72B LLM servers (TP=2, vLLM) ---
CUDA_VISIBLE_DEVICES=1,2 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port $LLM_PORT_1 \
    --max-model-len 32768 --trust-remote-code \
    --tensor-parallel-size 2 &

CUDA_VISIBLE_DEVICES=3,4 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port $LLM_PORT_2 \
    --max-model-len 32768 --trust-remote-code \
    --tensor-parallel-size 2 &

CUDA_VISIBLE_DEVICES=5,6 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --host 127.0.0.1 --port $LLM_PORT_3 \
    --max-model-len 32768 --trust-remote-code \
    --tensor-parallel-size 2 &

# --- Wait for all servers ---
echo "[INFO] Waiting for Embed (port $EMBED_PORT)..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done

for port in $LLM_PORT_1 $LLM_PORT_2 $LLM_PORT_3; do
    echo "[INFO] Waiting for LLM on port $port..."
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Port $port ready!" && break
        [ "$i" -eq 1800 ] && echo "[ERROR] Port $port timeout" && exit 1
        sleep 1
    done
done
echo "[INFO] All 4 vLLM servers ready."

# ============================================================================
# Helper: generate config
# ============================================================================
gen_config() {
    local llm_port=$1 experiment_name=$2 num_sections=$3
    local k_retrieve=$4 enable_vd=$5 build_strategy=$6
    local tau=${7:-0.62} weight_sim=${8:-0.5} weight_q=${9:-0.5}
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
  build_strategy: ${build_strategy}
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
  mode: train
  num_sections: ${num_sections}
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
  tau: ${tau}
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
  weight_sim: ${weight_sim}
  weight_q: ${weight_q}
CFGEOF
    echo "$cfg_path"
}

# ============================================================================
# Launch experiments
# ============================================================================

# --- 1. MemRL proc resume (GPU 1-2, from S9 b20) ---
(
    export MEMRL_RUN_ID="$MEMRL_PROC_RUN_ID"
    CFG=$(gen_config $LLM_PORT_1 "alfworld_memrl_qwen72b" 10 3 true proceduralization 0.62 0.5 0.5)
    echo "[EXP] 72B MemRL (proc) resume starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B MemRL (proc) resume exit=$? at $(date)"
) &
PID_MEMRL_PROC=$!

# --- 2. MemRL traj (GPU 3-4, from scratch) ---
(
    export MEMRL_RUN_ID="${TS//_/-}"
    CFG=$(gen_config $LLM_PORT_2 "alfworld_memrl_traj_qwen72b" 10 3 true trajectory 0.62 0.5 0.5)
    echo "[EXP] 72B MemRL (traj) starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B MemRL (traj) exit=$? at $(date)"
) &
PID_MEMRL_TRAJ=$!

# --- 3. RAG (GPU 5-6, from scratch) → then MemP → then Mem0 (sequential) ---
(
    export MEMRL_RUN_ID="${TS//_/-}-rag"

    # 3a. RAG
    CFG=$(gen_config $LLM_PORT_3 "alfworld_rag_qwen72b" 10 3 false trajectory 0.0 1.0 0.0)
    echo "[EXP] 72B RAG starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B RAG exit=$? at $(date)"

    # 3b. MemP (after RAG finishes)
    export MEMRL_RUN_ID="${TS//_/-}-memp"
    CFG=$(gen_config $LLM_PORT_3 "alfworld_memp_qwen72b" 10 3 false proceduralization 0.0 1.0 0.0)
    echo "[EXP] 72B MemP starting..."
    python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
    echo "[EXP] 72B MemP exit=$? at $(date)"

    # 3c. Mem0 (after MemP finishes)
    export MEMRL_RUN_ID="${TS//_/-}-mem0"
    export MEMRL_MEM0_LLM_BASE_URL="http://localhost:${LLM_PORT_3}/v1/"
    CFG=$(gen_config $LLM_PORT_3 "alfworld_mem0_qwen72b" 10 3 false proceduralization 0.0 1.0 0.0)
    echo "[EXP] 72B Mem0 starting..."
    python3 run/run_alfworld.py --config "$CFG" --mem0 --skip_initial_eval
    echo "[EXP] 72B Mem0 exit=$? at $(date)"
) &
PID_SEQ=$!

echo "[INFO] 3 parallel tracks launched."
echo "  Track 1 (GPU 1-2): MemRL proc resume PID=$PID_MEMRL_PROC"
echo "  Track 2 (GPU 3-4): MemRL traj PID=$PID_MEMRL_TRAJ"
echo "  Track 3 (GPU 5-6): RAG → MemP → Mem0 PID=$PID_SEQ"

wait $PID_MEMRL_PROC; echo "[DONE] MemRL proc resume: $?"
wait $PID_MEMRL_TRAJ; echo "[DONE] MemRL traj: $?"
wait $PID_SEQ; echo "[DONE] RAG → MemP → Mem0: $?"

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
