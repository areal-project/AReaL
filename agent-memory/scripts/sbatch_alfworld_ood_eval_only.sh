#!/bin/bash
#SBATCH --job-name=yl-alf-ood-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_ood_eval_%j.log
#SBATCH --error=logs/alf_ood_eval_%j.log

# Eval-only: fill in missing OOD (valid_unseen) for failure summary + plain MemRL
# S1, S3, S5 are missing OOD eval. Each eval ~15-30min.
# Total: 6 evals × ~20min = ~2-3h

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "OOD eval-only for holdout (failure_summary + memrl, S1/S3/S5)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_ood_eval_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
git checkout region-dev --quiet 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null; find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
rm -rf *.egg-info 2>/dev/null; pip uninstall -y memrl 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan vllm textworld alfworld --quiet 2>/dev/null
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 1200); do curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && break; sleep 1; done

# --- Start LLM vLLM ---
export NCCL_ASYNC_ERROR_HANDLING=1; export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$
start_llm() {
    [ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && sleep 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
}
start_llm
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready!' && break
    kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null || { echo "[ERROR] LLM died"; kill "$EMBED_PID" 2>/dev/null; exit 1; }
    [ "$i" -eq 1800 ] && { echo '[ERROR] LLM timeout'; exit 1; }
    sleep 1
done

# --- Watchdog ---
WATCHDOG_KEEP_RUNNING=/tmp/watchdog_keep_$$
touch "$WATCHDOG_KEEP_RUNNING"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP_RUNNING" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then fails=0
        else fails=$((fails+1)); [ "$fails" -ge 3 ] && { start_llm; sleep 300; fails=0; }; fi
        sleep 60
    done
) &
WD_PID=$!

# Ckpt paths
FAIL_CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/holdout/exp_alfworld_holdout_pickplace_failure_summary_qwen72b_2section_20260609-104715/local_cache"
MEMRL_CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/holdout/exp_alfworld_holdout_pick_and_place_simple_qwen72b_memrl_2section_20260606-180057/local_cache"

# Base config template (will be patched per eval)
BASE_YAML=$(cat << 'YAMLEOF'
llm:
  provider: "openai"
  api_key: "EMPTY"
  base_url: "http://localhost:LLM_PORT_PLACEHOLDER/v1/"
  model: "Qwen2.5-72B-Instruct"
  temperature: 0
  max_tokens: 4096
embedding:
  provider: "openai"
  api_key: "EMPTY"
  base_url: "http://localhost:EMBED_PORT_PLACEHOLDER/v1/"
  model: "Qwen/Qwen3-Embedding-8B"
  max_text_len: 8196
memory:
  build_strategy: "proceduralization"
  retrieve_strategy: "query"
  update_strategy: "adjustment"
  k_retrieve: 5
  max_keywords: 5
  add_similarity_threshold: 0.9
  memory_budget_tokens: 0
  sim_norm_mean: 0.5187
  sim_norm_std: 0.1203
environment:
  alfworld_config_path: "configs/envs/alfworld.yaml"
  alfworld_env_type: "AlfredTWEnv"
experiment:
  random_seed: 42
  enable_value_driven: true
  experiment_name: "EXPNAME_PLACEHOLDER"
  mode: "test"
  num_sections: 1
  batch_size: 32
  dataset_ratio: 1.0
  few_shot_path: "data/alfworld/alfworld_examples.json"
  baseline_mode: null
  baseline_k: 10
  output_dir: "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/holdout"
  max_steps: 30
  save_trajectories: false
  save_memories: false
  bon: 0
  valid_interval: 1
  test_interval: 1
  holdout_subtask: "alf/pick_and_place_simple"
  ckpt_resume_enabled: true
  ckpt_resume_path: "CKPT_PLACEHOLDER"
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
YAMLEOF
)

run_eval() {
    local METHOD="$1"   # "failure_summary" or "memrl"
    local SECTION="$2"  # 1, 3, or 5
    local CKPT_BASE="$3"

    local CKPT_PATH="${CKPT_BASE}/snapshot/${SECTION}"
    local EXP_NAME="ood_eval_${METHOD}_s${SECTION}"

    echo "=========================================="
    echo "[EVAL] ${METHOD} Section ${SECTION} — OOD eval-only"
    echo "  ckpt: ${CKPT_PATH}"
    echo "  Start: $(date)"
    echo "=========================================="

    if [ ! -d "$CKPT_PATH" ]; then
        echo "[SKIP] ckpt not found: $CKPT_PATH"
        return
    fi

    local CFG="/tmp/alf_ood_${METHOD}_s${SECTION}_$$.yaml"
    echo "$BASE_YAML" | \
        sed "s|LLM_PORT_PLACEHOLDER|${LLM_PORT}|g" | \
        sed "s|EMBED_PORT_PLACEHOLDER|${EMBED_PORT}|g" | \
        sed "s|EXPNAME_PLACEHOLDER|${EXP_NAME}|g" | \
        sed "s|CKPT_PLACEHOLDER|${CKPT_BASE}|g" > "$CFG"

    # Override ckpt_resume_path to point to the specific section snapshot
    # The runner will load this snapshot and run eval
    sed -i "s|ckpt_resume_path:.*|ckpt_resume_path: \"${CKPT_BASE}\"|" "$CFG"

    python run/run_alfworld.py --config "$CFG" \
        --holdout_subtask alf/pick_and_place_simple \
        --holdout_eval_pools train,valid

    echo "[EVAL] ${METHOD} S${SECTION} exit=$? at $(date)"
}

# Run all 6 evals
for SECTION in 1 3 5; do
    run_eval "failure_summary" "$SECTION" "$FAIL_CKPT"
done

for SECTION in 1 3 5; do
    run_eval "memrl" "$SECTION" "$MEMRL_CKPT"
done

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo "[INFO] All OOD evals done."
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $SINGULARITY_IMG bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"
rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
