#!/bin/bash
#SBATCH --job-name=yl-alf-qwen72b-nomem-then-resume
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_qwen72b_nomem_then_resume_%j.log
#SBATCH --error=logs/alf_qwen72b_nomem_then_resume_%j.log

# Combined sbatch: phase 1 runs Qwen2.5-72B no-mem eval on train+valid+test,
# then phase 2 resumes baseline 927704 from snapshot/2 and trains through section 10
# (num_sections=10). Sharing one vLLM saves ~30min of startup time vs submitting
# separately. Total wall time estimate: ~55h (phase 1 ~1.5h + phase 2 ~52h).

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld Qwen72B: NO-MEM eval (train+valid+test) THEN baseline resume snapshot/2 → s10"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_qwen72b_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

echo '[INFO] Switching to region-dev branch...'
git checkout region-dev --quiet 2>/dev/null || true
# git pull disabled: always use local working-tree code

echo '[INFO] Installing dependencies...'
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

NOMEM_CONFIG=/tmp/alf_qwen72b_nomem_config_$$.yaml
RESUME_CONFIG=/tmp/alf_qwen72b_resume_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_nomem_eval.yaml > "$NOMEM_CONFIG"
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_baseline_resume.yaml > "$RESUME_CONFIG"

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!

for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding vLLM is ready!' && break
    [ "$i" -eq 1200 ] && echo '[ERROR] Embedding vLLM failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Start LLM vLLM (Qwen2.5-72B-Instruct, TP=4, GPUs 0-3) ---
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$

start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
    echo "[WATCHDOG] Started vLLM pid=$(cat $LLM_PID_FILE) at $(date)"
}

start_llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM server is ready!' && break
    if ! kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null; then
        echo "[ERROR] LLM vLLM process died during startup (likely config error). Aborting."
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    if [ "$i" -eq 1800 ]; then
        echo '[ERROR] LLM vLLM failed to become ready within 1800s. Aborting.'
        kill "$(cat $LLM_PID_FILE)" 2>/dev/null
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Watchdog (covers BOTH phases) ---
WATCHDOG_KEEP_RUNNING=/tmp/watchdog_keep_$$
touch "$WATCHDOG_KEEP_RUNNING"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP_RUNNING" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            echo "[WATCHDOG] health check failed ($fails/3) at $(date)"
            if [ "$fails" -ge 3 ]; then
                echo "[WATCHDOG] restarting vLLM at $(date)"
                start_llm
                restarted=0
                for w in $(seq 1 300); do
                    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
                        echo "[WATCHDOG] vLLM restarted at $(date)"
                        restarted=1; break
                    fi
                    sleep 1
                done
                [ "$restarted" -eq 1 ] && fails=0 || echo "[WATCHDOG] restart FAILED at $(date)"
            fi
        fi
        sleep 60
    done
) &
WD_PID=$!

# ============ PHASE 1: NO-MEM EVAL ============
echo '=========================================='
echo '[PHASE 1] ALFWorld Qwen72B NO-MEM eval on train + valid + test'
echo 'k_retrieve=0, mode=test, --eval_train (train+valid+test)'
echo '=========================================='

python run/run_alfworld.py --config "$NOMEM_CONFIG" --eval_train
NOMEM_EXIT=$?
echo "[INFO] Phase 1 (no-mem eval) exited with code: $NOMEM_EXIT"

# Don't fail the whole job if phase 1 failed — still try resume
if [ "$NOMEM_EXIT" -ne 0 ]; then
    echo "[WARNING] Phase 1 failed (exit $NOMEM_EXIT) — proceeding to phase 2 anyway"
fi

# ============ PHASE 2: BASELINE RESUME ============
echo '=========================================='
echo '[PHASE 2] ALFWorld Qwen72B baseline RESUME from snapshot/2 → train through section 10'
echo 'k_retrieve=5, mode=train, ckpt_resume=snapshot/2, num_sections=10 (8 new sections)'
echo '=========================================='

python run/run_alfworld.py --config "$RESUME_CONFIG" &
MAIN_PID=$!
wait "$MAIN_PID"; RESUME_EXIT=$?
echo "[INFO] Phase 2 (baseline resume) exited with code: $RESUME_EXIT"

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] Done. Phase 1 exit=' "$NOMEM_EXIT" ' Phase 2 exit=' "$RESUME_EXIT"
INNEREOF

chmod +x "$INNER_SCRIPT"

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"

rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
