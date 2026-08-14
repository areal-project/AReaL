#!/bin/bash
#SBATCH --job-name=yl-alf-combined
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_combined_%j.log
#SBATCH --error=logs/alf_combined_%j.log

# Combined 8-GPU job:
#   GPU 0:   Embedding (shared, Qwen3-Embedding-8B)
#   GPU 1:   Qwen3.6 — RAG baseline
#   GPU 2:   Qwen3.6 — Self-RAG baseline
#   GPU 3:   Qwen3.6 — Mem0 baseline
#   GPU 4-7: Qwen2.5-72B TP=4 — Region+FS (bs=64, paper rerun)

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_72B="/storage/openpsi/models/Qwen2.5-72B-Instruct"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

EMBED_PORT=8401
LLM_72B_PORT=8300
RAG_PORT=8402
SELFRAG_PORT=8404
MEM0_PORT=8406

echo "=========================================="
echo "ALFWorld Combined: 3x Qwen3.6 baselines + 72B Region+FS"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

###############################################################################
# Phase 1: Start ALL vLLM servers (one Singularity container, 5 processes)
###############################################################################
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --port $EMBED_PORT --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_72B --served-model-name Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port $LLM_72B_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $RAG_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --disable-log-requests --seed 42 &

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $SELFRAG_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --disable-log-requests --seed 42 &

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $MEM0_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --disable-log-requests --seed 42 &

wait
" &
VLLM_BG_PID=$!

###############################################################################
# Phase 2: Wait for all servers to be healthy
###############################################################################
wait_for_server() {
    local name=$1 port=$2 timeout=$3
    echo "[INFO] Waiting for $name (port $port)..."
    for i in $(seq 1 $timeout); do
        curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && echo "[INFO] $name ready!" && return 0
        kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM container died while waiting for $name"; exit 1; }
        [ "$i" -eq "$timeout" ] && echo "[ERROR] $name timeout after ${timeout}s" && kill $VLLM_BG_PID 2>/dev/null && exit 1
        sleep 1
    done
}

wait_for_server "Embedding"        $EMBED_PORT   1200
wait_for_server "72B-LLM"          $LLM_72B_PORT 1800
wait_for_server "Qwen3.6-RAG"      $RAG_PORT     1800
wait_for_server "Qwen3.6-SelfRAG"  $SELFRAG_PORT 1800
wait_for_server "Qwen3.6-Mem0"     $MEM0_PORT    1800

echo "=========================================="
echo "[INFO] All 5 vLLM servers ready. Starting experiments..."
echo "=========================================="

###############################################################################
# Phase 3: Write inline configs and launch runners
###############################################################################
INNER_SCRIPT=$(mktemp /tmp/alf_combined_runner_XXXXXX.sh)
cat > "$INNER_SCRIPT" << INNEREOF
#!/bin/bash
MEMRL_DIR="$MEMRL_DIR"
EMBED_PORT=$EMBED_PORT
LLM_72B_PORT=$LLM_72B_PORT
RAG_PORT=$RAG_PORT
SELFRAG_PORT=$SELFRAG_PORT
MEM0_PORT=$MEM0_PORT

cd "\$MEMRL_DIR"
echo "[INFO] Installing runner deps..."
find . -name "*.pyc" -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
pip install -e /storage/openpsi/users/yl/agent-memory/mem0 --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Write RAG config ---
# RAG stores raw trajectories (no LLM proceduralization), retrieves by pure sim.
cat > /tmp/rag_qwen36.yaml << EOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${RAG_PORT}/v1/"
  model: Qwen3.6-35B-A3B
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${EMBED_PORT}/v1/"
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
  experiment_name: alfworld_rag_qwen36
  mode: train
  num_sections: 10
  batch_size: 32
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
EOF

# --- Write Self-RAG config ---
# Self-RAG = RAG + LLM critique. Same storage as RAG (trajectory), adds reflection filtering.
cat > /tmp/selfrag_qwen36.yaml << EOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${SELFRAG_PORT}/v1/"
  model: Qwen3.6-35B-A3B
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${EMBED_PORT}/v1/"
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
  experiment_name: alfworld_selfrag_qwen36
  mode: train
  num_sections: 10
  batch_size: 32
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: reflection
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
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
EOF

# --- Write Mem0 config ---
# Mem0 uses Mem0MemoryService (--mem0 flag); build_strategy is unused but set for consistency.
cat > /tmp/mem0_qwen36.yaml << EOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${MEM0_PORT}/v1/"
  model: Qwen3.6-35B-A3B
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: "http://localhost:\${EMBED_PORT}/v1/"
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
  experiment_name: alfworld_mem0_qwen36
  mode: train
  num_sections: 10
  batch_size: 32
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
EOF

# --- Launch all 4 runners in parallel ---
echo "[1/4] Starting RAG (Qwen3.6)..."
python run/run_alfworld.py --config /tmp/rag_qwen36.yaml --skip_initial_eval &
RAG_PID=\$!

echo "[2/4] Starting Self-RAG (Qwen3.6)..."
python run/run_alfworld.py --config /tmp/selfrag_qwen36.yaml --skip_initial_eval &
SELFRAG_PID=\$!

echo "[3/4] Starting Mem0 (Qwen3.6)..."
python run/run_alfworld.py --config /tmp/mem0_qwen36.yaml --mem0 --skip_initial_eval &
MEM0_PID=\$!

echo "[4/4] Starting Region+FS (72B, bs=64)..."
python run/run_alfworld.py \
    --config configs/rl_alf_config.qwen72b_region_10section_nozn_bs64.yaml \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --val_lambda_max 0.05 \
    --no_z_norm \
    --explore_schedule "0,2,2,1,1,1,1,0,0,0" \
    --failure_summary_n_slots 2 \
    --skip_initial_eval &
REGION_PID=\$!

echo "=========================================="
echo "All 4 runners launched. PIDs:"
echo "  RAG=\$RAG_PID  Self-RAG=\$SELFRAG_PID  Mem0=\$MEM0_PID  Region=\$REGION_PID"
echo "=========================================="

wait \$RAG_PID;     echo "[RAG]      exit=\$? at \$(date)"
wait \$SELFRAG_PID; echo "[Self-RAG] exit=\$? at \$(date)"
wait \$MEM0_PID;    echo "[Mem0]     exit=\$? at \$(date)"
wait \$REGION_PID;  echo "[Region]   exit=\$? at \$(date)"

echo "=========================================="
echo "All experiments done. \$(date)"
echo "=========================================="
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --no-home --writable-tmpfs --bind /storage:/storage \
    $RUNNER_IMG bash "$INNER_SCRIPT"
rm -f "$INNER_SCRIPT"

# Cleanup
kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)  Exit: $?"
