#!/bin/bash
#SBATCH --job-name=yl-bcb-dsv3-nomem-memrl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/bcb_dsv3_nomem_memrl_%j.log
#SBATCH --error=logs/bcb_dsv3_nomem_memrl_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-v3.2"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))

echo "=========================================="
echo "BCB DeepSeek-V3: NoMem → MemRL baseline (TP=8, embedding via API)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_dsv3_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"
cd "$MEMRL_DIR"

LLM_PID=""
TEMP_CONFIG=""
cleanup() {
    [ -n "$LLM_PID" ] && kill "$LLM_PID" 2>/dev/null || true
    [ -n "$LLM_PID" ] && wait "$LLM_PID" 2>/dev/null || true
    [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG" 2>/dev/null || true
}
trap 'trap - EXIT; cleanup; exit 143' SIGINT SIGTERM
trap cleanup EXIT

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . 2>&1 | tail -3
python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo "[FATAL] memrl import failed"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan 2>&1 | tail -3
python -c 'import memos, mem0' || { echo "[FATAL] memos/mem0 import failed"; exit 1; }
pip install vllm 2>&1 | tail -3
python -c 'import vllm; print("[OK] vllm:", vllm.__version__)' || { echo "[FATAL] vllm import failed"; exit 1; }
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt 2>&1 | tail -3
echo '[INFO] Installing critical eval-time modules...'
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium 2>&1 | tail -5
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt 2>&1 | tail -5 || true
echo '[INFO] All dependencies installed.'
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/bcb_dsv3_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g" \
    configs/rl_bcb_config.deepseek_v3.yaml > "$TEMP_CONFIG"

# --- Start DeepSeek-V3 vLLM (TP=8, all GPUs) ---
echo '[INFO] Starting DeepSeek-V3 (TP=8, 8x GPU)...'
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name deepseek-v3.2 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --tokenizer-mode deepseek_v32 \
    --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 &
LLM_PID=$!
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready' && break
    [ "$i" -eq 1800 ] && echo '[ERROR] LLM failed to start in 30min' && exit 1
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
    --checkpoint_interval 100 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_nomem
echo "[INFO] No-Memory exit code: $?"

# ==== Phase 2: MemRL baseline (3 epochs, retrieve_k=10) ====
echo '=========================================='
echo '[Phase 2] MemRL baseline (retrieve_k=10, 3 epochs)'
echo '=========================================='
python run/run_bcb.py \
    --config "$TEMP_CONFIG" \
    --split instruct --subset full --epochs 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_memrl

EXIT_CODE=$?
echo "[INFO] MemRL exit code: $EXIT_CODE"
echo '[INFO] Done.'
exit $EXIT_CODE
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT"
RC=$?
rm -f "$INNER_SCRIPT"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $RC"
echo "=========================================="
exit $RC
