#!/bin/bash
#SBATCH --job-name=yl-alf-llama70b-region
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_llama70b_region_%j.log
#SBATCH --error=logs/alf_llama70b_region_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Meta__Llama-3.3-70B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld Region: Llama-3.3-70B-Instruct (TP=4, native 128K ctx) + Qwen3-Embedding-8B (local)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_llama70b_XXXXXX.sh)
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

TEMP_CONFIG=/tmp/alf_llama70b_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" configs/rl_alf_config.llama70b_region.yaml > "$TEMP_CONFIG"

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

# --- Start LLM vLLM (Llama-3.3-70B-Instruct, TP=4, GPUs 0-3) ---
# Llama-3.3-70B-Instruct has native max_position_embeddings=131072 with llama3 RoPE scaling baked in.
# No --rope-scaling override needed.
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
        --model "$MODEL_PATH" --served-model-name Llama-3.3-70B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 131072 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
    echo "[WATCHDOG] Started vLLM pid=$(cat $LLM_PID_FILE) at $(date)"
}

start_llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM server is ready!' && break
    # If vLLM process already died (e.g. config validation error), fail fast — watchdog won't help.
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

# --- Run experiment ---
echo '=========================================='
echo '[ALFWorld-Llama70B-Region] Llama-3.3-70B-Instruct + Qwen3-Embedding-8B (local)'
echo 'Task types: 7 natural subtasks, max_steps=30'
echo '=========================================='

python run/run_alfworld.py \
    --config "$TEMP_CONFIG" --region --region_gating_mode multiplicative --skip_initial_eval &
MAIN_PID=$!

# --- Watchdog ---
(
    fails=0
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            echo "[WATCHDOG] health check failed ($fails/3) at $(date)"
            if [ "$fails" -ge 3 ]; then
                echo "[WATCHDOG] restarting vLLM at $(date)"
                start_llm
                restarted=0
                for w in $(seq 1 600); do
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

wait "$MAIN_PID"; ALF_EXIT=$?
echo "[INFO] ALFWorld Llama70B region exited with code: $ALF_EXIT"

kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] Done.'
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
