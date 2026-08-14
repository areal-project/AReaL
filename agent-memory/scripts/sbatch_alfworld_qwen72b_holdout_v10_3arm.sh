#!/bin/bash
#SBATCH --job-name=yl-alf-v10-3arm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_v10_3arm_%j.log
#SBATCH --error=logs/alf_v10_3arm_%j.log

# v10 holdout 3-arm A/B: H (Hybrid) → P (Pure-D1) → S (Sim*D1)
# All three arms share a single vLLM instance to save ~1h startup time
# (~30min startup × 2 saved restarts).
#
# See docs/ALFWORLD_V10_HOLDOUT_IMPL.md for design + offline validation.
#
# Expected wall time:
#   Phase 0 (vLLM warmup):     ~30min
#   Phase 1 (Arm H, main):     ~10h
#   Phase 2 (Arm P):           ~8h  (no sim retrieval, slightly faster)
#   Phase 3 (Arm S):           ~10h
#   Total:                     ~30h
#
# Baselines for comparison (already done, no need to rerun):
#   929119 plain MemRL  S2 holdout SR = 64.73%
#   927950 region xfer  S2 holdout SR = 65.45%
#
# v10 targets:
#   Arm H (main): SR >= 67%  (Codex predicts 67-70%)
#   Arm P:        SR >= 67%  (Codex predicts 67-69%)
#   Arm S:        SR >= 65%  (control; sim+D1 vs plain sim)

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld v10 3-arm holdout (pick_and_place_simple): H → P → S"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_v10_XXXXXX.sh)
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

# Patch all three configs with current job's ports
HYBRID_CONFIG=/tmp/alf_v10_hybrid_$$.yaml
PURE_D1_CONFIG=/tmp/alf_v10_pure_d1_$$.yaml
SIM_D1_CONFIG=/tmp/alf_v10_sim_d1_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_holdout_pickplace_v10_hybrid_2section.yaml > "$HYBRID_CONFIG"
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_holdout_pickplace_v10_pure_d1_2section.yaml > "$PURE_D1_CONFIG"
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_holdout_pickplace_v10_sim_d1_2section.yaml > "$SIM_D1_CONFIG"

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

# --- Watchdog (covers ALL 3 phases) ---
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

# ============ PHASE 1: ARM H (Hybrid) — MAIN RECOMMENDATION ============
echo '=========================================='
echo '[PHASE 1] Arm H (Hybrid): top-3 D1 anchors + top-2 sim*D1 (pool 500)'
echo "Start: $(date)"
echo '=========================================='

python run/run_alfworld.py \
    --config "$HYBRID_CONFIG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval
H_EXIT=$?
echo "[INFO] Phase 1 (Arm H) exited with code: $H_EXIT at $(date)"

# ============ PHASE 2: ARM P (Pure-D1) ============
echo '=========================================='
echo '[PHASE 2] Arm P (Pure-D1): fixed top-5 from highest-D1 regions (no sim)'
echo "Start: $(date)"
echo '=========================================='

python run/run_alfworld.py \
    --config "$PURE_D1_CONFIG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval
P_EXIT=$?
echo "[INFO] Phase 2 (Arm P) exited with code: $P_EXIT at $(date)"

# ============ PHASE 3: ARM S (Sim*D1) — CONTROL ============
echo '=========================================='
echo '[PHASE 3] Arm S (Sim*D1): all top-5 by sim*D1 from pool 500'
echo "Start: $(date)"
echo '=========================================='

python run/run_alfworld.py \
    --config "$SIM_D1_CONFIG" --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval
S_EXIT=$?
echo "[INFO] Phase 3 (Arm S) exited with code: $S_EXIT at $(date)"

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] Done. Exit codes: H='"$H_EXIT"' P='"$P_EXIT"' S='"$S_EXIT"
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
