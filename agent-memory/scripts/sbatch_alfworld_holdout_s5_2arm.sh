#!/bin/bash
#SBATCH --job-name=yl-alf-holdout-s5-2arm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_holdout_s5_2arm_%j.log
#SBATCH --error=logs/alf_holdout_s5_2arm_%j.log

# 2-arm holdout resume to 5 sections:
#   Phase 1: Region failure summary (resume S2 → S3/S4/S5)
#   Phase 2: Plain MemRL (resume S3 → S4/S5)
# Shared vLLM.
# Expected: ~15h (failure summary 3 sections + memrl 2 sections)

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "Holdout 5-section 2-arm: failure_summary (S3-S5) + plain MemRL (S4-S5)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_s5_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

git checkout region-dev --quiet 2>/dev/null || true
# git pull disabled: always use local working-tree code

find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
pip install textworld alfworld --quiet 2>/dev/null || true

python -c "import memrl; print('[VERIFY] memrl loaded from:', memrl.__file__)"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

FAIL_CONFIG=/tmp/alf_fail_s5_$$.yaml
MEMRL_CONFIG=/tmp/alf_memrl_s5_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_holdout_pickplace_failure_summary_5section_resume.yaml > "$FAIL_CONFIG"
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_holdout_pickplace_memrl_5section_resume.yaml > "$MEMRL_CONFIG"

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding ready!' && break
    [ "$i" -eq 1200 ] && echo '[ERROR] Embedding failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Start LLM vLLM ---
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$
start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        kill -0 "$old_pid" 2>/dev/null && kill "$old_pid" 2>/dev/null && sleep 2
    fi
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
    [ "$i" -eq 1800 ] && { echo '[ERROR] LLM timeout'; kill "$(cat $LLM_PID_FILE)" "$EMBED_PID" 2>/dev/null; exit 1; }
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

# ============ PHASE 1: Region Failure Summary (resume S2 → S3/S4/S5) ============
echo '=========================================='
echo '[PHASE 1] Region failure summary: resume S2 → S3/S4/S5'
echo "Start: $(date)"
echo '=========================================='

python run/run_alfworld.py \
    --config "$FAIL_CONFIG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval \
    --failure_summary_n_slots 2 \
    --failure_summary_path analysis/region_failure_summaries.json
FAIL_EXIT=$?
echo "[INFO] Phase 1 (failure summary) exited: $FAIL_EXIT at $(date)"

# ============ PHASE 2: Plain MemRL (resume S3 → S4/S5) ============
echo '=========================================='
echo '[PHASE 2] Plain MemRL: resume S3 → S4/S5'
echo "Start: $(date)"
echo '=========================================='

python run/run_alfworld.py \
    --config "$MEMRL_CONFIG" \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval
MEMRL_EXIT=$?
echo "[INFO] Phase 2 (plain MemRL) exited: $MEMRL_EXIT at $(date)"

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo "[INFO] Done. failure_summary=$FAIL_EXIT, memrl=$MEMRL_EXIT"
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"
rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
