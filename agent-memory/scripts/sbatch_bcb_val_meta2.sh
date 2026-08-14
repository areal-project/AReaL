#!/bin/bash
#SBATCH --job-name=yl-bcb-val-meta2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --exclude=slurmd-41
#SBATCH --output=logs/bcb_val_meta2_%j.log
#SBATCH --error=logs/bcb_val_meta2_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

# 927120 Region Domain E3 checkpoint (best val=44.4%)
CHECKPOINT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/training/deepseek_old_region_domain_add_5gpu/bigcodebench_eval/instruct_full/region/20260530_203004_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch3/snapshot/3"

echo "=========================================="
echo "BCB Val-Only: Meta Header v2 on 927120 E3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_val_meta2_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"; CHECKPOINT="$6"
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
pip install --no-cache-dir -e . 2>&1 | tail -3
python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo "[FATAL] memrl import failed"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan 2>&1 | tail -3
python -c 'import memos, mem0' || { echo "[FATAL] memos/mem0 import failed"; exit 1; }
pip install vllm 2>&1 | tail -3
python -c 'import vllm; print("[OK] vllm:", vllm.__version__)' || { echo "[FATAL] vllm import failed"; exit 1; }
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt 2>&1 | tail -3
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium 2>&1 | tail -5
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt 2>&1 | tail -5 || true
python -c "
import importlib, sys
required = ['faker', 'statsmodels', 'xlwt', 'docx', 'sendgrid', 'openpyxl', 'xlrd', 'seaborn']
missing = [m for m in required if not importlib.util.find_spec(m)]
if missing: print(f'[FATAL] missing: {missing}'); sys.exit(1)
" || { echo "[FATAL] critical eval modules missing"; exit 1; }
echo '[INFO] All dependencies verified.'
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/bcb_val_meta2_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_bcb_config.deepseek_old_region_bs8.yaml > "$TEMP_CONFIG"

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
    --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 131072 --gpu-memory-utilization 0.80 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
LLM_PID=$!
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 900); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready' && break
    [ "$i" -eq 900 ] && echo '[ERROR] LLM failed' && exit 1
    sleep 1
done

OUTPUT_BASE="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/training/deepseek_old_region_domain_add_5gpu/val_meta2_eval"

# Use ORIGINAL 927120 region config: EMA utility, temp=1.0, shrinkage_top_n=3, min_cluster_size=5
COMMON_ARGS="--config $TEMP_CONFIG --split instruct --subset full --epochs 1 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --strip_think --task_cluster_k 0 \
    --region_gating_mode additive --region_utility_mode ema \
    --region_temperature 1.0 --shrinkage_top_n 3 \
    --propagation_eta 0.03 --propagation_k 30 --propagation_sim_min 0.40 \
    --resume_from $CHECKPOINT --eval_only"

echo '=========================================='
echo 'Run 1: Meta Header v2 (improved instructions)'
echo '=========================================='
python run/run_bcb_region.py $COMMON_ARGS \
    --region_meta_header \
    --output_dir "$OUTPUT_BASE/meta_header_v2"
echo "[INFO] Meta header v2 exit code: $?"

echo '=========================================='
echo 'Run 2: No header, no gate (plain region retrieval)'
echo '=========================================='
python run/run_bcb_region.py $COMMON_ARGS \
    --output_dir "$OUTPUT_BASE/plain_region"
echo "[INFO] Plain region exit code: $?"

echo '=========================================='
echo 'Run 3: No-mem baseline'
echo '=========================================='
python run/run_bcb_region.py $COMMON_ARGS \
    --retrieval_mode no_mem \
    --output_dir "$OUTPUT_BASE/no_mem"
echo "[INFO] No-mem exit code: $?"

echo '=========================================='
echo "End time: $(date)"
echo '=========================================='
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT" "$CHECKPOINT"
RC=$?
rm -f "$INNER_SCRIPT"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $RC"
echo "=========================================="
exit $RC
