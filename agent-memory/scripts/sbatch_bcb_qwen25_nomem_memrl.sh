#!/bin/bash
#SBATCH --job-name=yl-bcb-qwen25-nomem-memrl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_qwen25_nomem_memrl_%j.log
#SBATCH --error=logs/bcb_qwen25_nomem_memrl_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen__Qwen2.5-32B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "BCB Qwen2.5-32B-Instruct: NoMem → MemRL baseline (serial)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_qwen25_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"

EMBED_PID=""
LLM_PID=""
TEMP_CONFIG=""
cleanup() {
    [ -n "$LLM_PID" ]   && kill "$LLM_PID"   2>/dev/null || true
    [ -n "$EMBED_PID" ] && kill "$EMBED_PID" 2>/dev/null || true
    [ -n "$LLM_PID" ]   && wait "$LLM_PID"   2>/dev/null || true
    [ -n "$EMBED_PID" ] && wait "$EMBED_PID" 2>/dev/null || true
    [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG" 2>/dev/null || true
}
trap 'trap - EXIT; cleanup; exit 143' SIGINT SIGTERM
trap cleanup EXIT

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true

python -c 'import memrl; print(memrl.__file__)'
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/bcb_qwen25_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_bcb_config.qwen25_32b_instruct.yaml > "$TEMP_CONFIG"

# Override Qwen2.5-32B-Instruct's generation_config.json which sets temperature=0.7,
# top_k=20, top_p=0.8, repetition_penalty=1.05 as defaults. vLLM uses these as
# "default chat sampling params" even when our API request explicitly passes temperature=0.
# Fix: write a neutral generation_config.json so vLLM doesn't override our params.
echo '{"do_sample": false}' > "$MODEL_PATH/generation_config.json"

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 600); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding ready' && break
    [ "$i" -eq 600 ] && echo '[ERROR] Embedding failed' && exit 1
    sleep 1
done

# --- Start LLM vLLM (TP=4, GPUs 0-3) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
LLM_PID=$!
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 900); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready' && break
    [ "$i" -eq 900 ] && echo '[ERROR] LLM failed' && exit 1
    sleep 1
done

# ==== Phase 1: No-Memory baseline (1 epoch, retrieve_k=0) ====
echo '=========================================='
echo '[Phase 1] No-Memory baseline (retrieve_k=0, 1 epoch)'
echo '=========================================='
python run/run_bcb.py \
    --config "$TEMP_CONFIG" \
    --split instruct --subset full --epochs 1 \
    --retrieve_k 0 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --strip_think \
    --checkpoint_interval 50 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/qwen25_32b_nomem
echo "[INFO] No-Memory exit code: $?"

# ==== Phase 2: MemRL baseline (3 epochs, retrieve_k=10) ====
echo '=========================================='
echo '[Phase 2] MemRL baseline (retrieve_k=10, 3 epochs)'
echo '=========================================='
python run/run_bcb.py \
    --config "$TEMP_CONFIG" \
    --split instruct --subset full --epochs 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --strip_think \
    --checkpoint_interval 50 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/qwen25_32b_memrl

EXIT_CODE=$?
echo "[INFO] MemRL exit code: $EXIT_CODE"
echo '[INFO] Done.'
exit $EXIT_CODE
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"
RC=$?
rm -f "$INNER_SCRIPT"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $RC"
echo "=========================================="
exit $RC
