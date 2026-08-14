#!/bin/bash
# ============================================================================
# AIStudio: Qwen2.5-72B ALFWorld Region+FS clean full train v3 (3x H200)
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B
#   GPU 1-2: Qwen2.5-72B TP=2 → Region+FS clean training
# ============================================================================
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen72b_regionfs_clean_v3_${1:-$(date +%Y%m%d_%H%M%S)}.log"

EMBED_PORT="${EMBED_PORT:-19070}"
LLM_PORT_1="${LLM_PORT:-19270}"
NCCL_PORT="${NCCL_PORT:-29670}"

export MEMRL_RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
MEMRL_RUN_ID="${MEMRL_RUN_ID//_/-}"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "AIStudio: Qwen2.5-72B Region+FS clean full train v3 (3 GPU)"
echo "Start: $(date)"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

cd "$MEMRL_DIR"
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --context-length 8192 --trust-remote-code \
    --is-embedding &
EMBED_PID=$!

CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path $QWEN72B_PATH --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port $LLM_PORT_1 \
    --trust-remote-code --context-length 32768 \
    --nccl-port "$NCCL_PORT" &
LLM_PID=$!

cleanup() {
    kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[INFO] Waiting for Embed..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done
echo "[INFO] Waiting for LLM..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT_1}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] LLM ready!" && break
    [ "$i" -eq 1800 ] && echo "[ERROR] LLM timeout" && exit 1
    sleep 1
done

cat > /tmp/alf_regionfs_clean_v3_$$.yaml << CFGEOF
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
  experiment_name: alfworld_regionfs_qwen72b_clean_v3
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
  tau: 0.60
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
  weight_sim: 0.45
  weight_q: 0.55
CFGEOF

echo "[EXP] 72B Region+FS clean full train v3 starting..."
echo "[EXP] Params: val_lambda_max=0.45, explore='0,1,1,1,0,0,0,0,0,0', shrinkage_k=2.5, tau=0.60, wq=0.55, ws=0.45, FS slots=1"
python3 run/run_alfworld.py \
    --config /tmp/alf_regionfs_clean_v3_$$.yaml \
    --region --region_gating_mode additive \
    --region_utility_mode beta \
    --shrinkage_confidence_k 2.5 --propagation_eta 0.12 \
    --val_lambda_max 0.45 --no_z_norm \
    --explore_schedule '0,1,1,1,0,0,0,0,0,0' \
    --failure_summary_n_slots 1
echo "[EXP] 72B Region+FS clean full train v3 exit=$? at $(date)"
