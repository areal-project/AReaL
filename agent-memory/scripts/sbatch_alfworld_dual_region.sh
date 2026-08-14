#!/bin/bash
#SBATCH --job-name=yl-alf-dual-region
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --gres=gpu:6
#SBATCH --output=logs/alf_dual_region_%j.log
#SBATCH --error=logs/alf_dual_region_%j.log

# Machine B: region+FS for both DSv4-Flash and Qwen3.6
# GPU layout: 0-3 DSv4-Flash TP=4 | 4-5 Qwen3.6 TP=2 | 6 Embed | 7 idle
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
DSV4_PATH="/storage/openpsi/models/deepseek-v4-flash"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

DSV4_PORT=8000
QWEN36_PORT=8100
EMBED_PORT=8001

echo "=========================================="
echo "ALFWorld Dual Model: Region+FS"
echo "DSv4-Flash (TP=4, GPU 0-3) + Qwen3.6 (TP=2, GPU 4-5) + Embed (GPU 6)"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_dual_region_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; DSV4_PATH="$2"; QWEN36_PATH="$3"; EMBED_PATH="$4"
DSV4_PORT="$5"; QWEN36_PORT="$6"; EMBED_PORT="$7"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

find . -name "*.pyc" -delete 2>/dev/null; find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan vllm textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start Embedding Server (GPU 6) ---
CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 600); do curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Embed ready" && break; sleep 1; done

# --- Start DSv4-Flash (GPU 0-3, TP=4) ---
export NCCL_ASYNC_ERROR_HANDLING=1; export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$DSV4_PATH" --served-model-name deepseek-v4-flash \
    --tensor-parallel-size 4 --port "$DSV4_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 --enable-expert-parallel \
    --kv-cache-dtype fp8 --block-size 256 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
DSV4_PID=$!

# --- Start Qwen3.6 (GPU 4-5, TP=2) ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port "$QWEN36_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --disable-log-requests --seed 42 &
QWEN36_PID=$!

export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# Wait for DSv4-Flash
for i in $(seq 1 2400); do
    curl -s "http://localhost:${DSV4_PORT}/health" > /dev/null 2>&1 && echo "[INFO] DSv4-Flash ready!" && break
    kill -0 $DSV4_PID 2>/dev/null || { echo "[ERROR] DSv4-Flash died"; break; }
    [ "$i" -eq 2400 ] && echo "[ERROR] DSv4-Flash timeout"
    sleep 1
done

# Wait for Qwen3.6
for i in $(seq 1 1800); do
    curl -s "http://localhost:${QWEN36_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Qwen3.6 ready!" && break
    kill -0 $QWEN36_PID 2>/dev/null || { echo "[ERROR] Qwen3.6 died"; break; }
    [ "$i" -eq 1800 ] && echo "[ERROR] Qwen3.6 timeout"
    sleep 1
done

echo "=========================================="
echo "All servers ready. Starting region+FS experiments..."
echo "=========================================="

REGION_FLAGS="--region --region_gating_mode additive --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.05 --no_z_norm --failure_summary_n_slots 2"
EXPLORE="--explore_schedule 0,2,2,1,1,1,1,0,0,0"

# --- DSv4-Flash: region+FS (background) ---
(
    echo "[DSv4] Region+FS (10 sections)"
    python run/run_alfworld.py \
        --config configs/rl_alf_config.dsv4flash_memrl.yaml \
        $REGION_FLAGS $EXPLORE --skip_initial_eval
    echo "[DSv4 region] exit=$? at $(date)"
) &
DSV4_EXP_PID=$!

# --- Qwen3.6: region+FS (background) ---
(
    echo "[Qwen3.6] Region+FS (10 sections)"
    python run/run_alfworld.py \
        --config configs/rl_alf_config.qwen36_memrl.yaml \
        $REGION_FLAGS $EXPLORE --skip_initial_eval
    echo "[Qwen3.6 region] exit=$? at $(date)"
) &
QWEN36_EXP_PID=$!

# Wait for both experiments to complete
wait $DSV4_EXP_PID; DSV4_EXIT=$?
wait $QWEN36_EXP_PID; QWEN36_EXIT=$?

echo "=========================================="
echo "All experiments done. DSv4=$DSV4_EXIT Qwen3.6=$QWEN36_EXIT"
echo "End: $(date)"
echo "=========================================="

kill $DSV4_PID $QWEN36_PID $EMBED_PID 2>/dev/null
wait $DSV4_PID $QWEN36_PID $EMBED_PID 2>/dev/null
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$DSV4_PATH" "$QWEN36_PATH" "$EMBED_PATH" "$DSV4_PORT" "$QWEN36_PORT" "$EMBED_PORT"
rm -f "$INNER_SCRIPT"
echo "End: $(date)"
