#!/bin/bash
#SBATCH --job-name=yl-bcb-region-failsum
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_region_failsum_%j.log
#SBATCH --error=logs/bcb_region_failsum_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "BCB Region + Failure Summary (10 epochs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_region_failsum_XXXXXX.sh)
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
# memrl editable install: must succeed
pip install --no-cache-dir -e . 2>&1 | tail -3
python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo "[FATAL] memrl import failed"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
# Core deps: must succeed
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan 2>&1 | tail -3
python -c 'import memos, mem0' || { echo "[FATAL] memos/mem0 import failed"; exit 1; }
pip install vllm 2>&1 | tail -3
python -c 'import vllm; print("[OK] vllm:", vllm.__version__)' || { echo "[FATAL] vllm import failed"; exit 1; }
# BCB main requirements
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt 2>&1 | tail -3
# BCB eval requirements: requirements-eval.txt pins numba==0.55.0 which is incompatible
# with Python 3.12 in this container (numba 0.55 requires Python <3.11). pip install -r
# fails midway and silently drops faker/statsmodels/xlwt/etc — those modules are actually
# needed at task eval time. Install the critical eval modules DIRECTLY (skipping numba).
echo '[INFO] Installing critical eval-time modules directly (bypass requirements-eval numba conflict)...'
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium 2>&1 | tail -5
# Then try the full requirements-eval (numba step will fail but most other modules will install)
echo '[INFO] Installing remaining BCB eval requirements (numba will fail, others should succeed)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt 2>&1 | tail -5 || true
# Verify critical eval-time modules (matches 6.3pp of missing-module FAILs in 930415):
echo '[INFO] Verifying critical eval-time modules...'
python -c "
import importlib, sys
required = ['faker', 'statsmodels', 'xlwt', 'docx', 'sendgrid', 'openpyxl', 'xlrd', 'seaborn']
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
        print(f'  [OK] {mod}')
    except ImportError as e:
        print(f'  [MISSING] {mod}: {e}')
        missing.append(mod)
if missing:
    print(f'[FATAL] {len(missing)} required eval modules missing: {missing}')
    sys.exit(1)
" || { echo "[FATAL] critical eval modules missing"; exit 1; }
echo '[INFO] All dependencies verified.'
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/bcb_region_failsum_config_$$.yaml
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

echo '=========================================='
echo 'Running Region + Failure Summary (10 epochs, DeepSeek-R1-Distill-Qwen-32B)'
echo '=========================================='

python run/run_bcb_region.py \
    --config "$TEMP_CONFIG" \
    --split instruct --subset full --epochs 10 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --strip_think --task_cluster_k 0 \
    --region_gating_mode additive --region_utility_mode beta \
    --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 15 \
    --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 \
    --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 \
    --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 \
    --failure_summary_n_slots 1 \
    --resume_from /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/training/deepseek_region_failure_summary_5gpu/bigcodebench_eval/instruct_full/region/20260610_150826_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch5/snapshot/step_704 \
    --resume_epoch 5 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/training/deepseek_region_failure_summary_5gpu

EXIT_CODE=$?
echo "[INFO] Region + Failure Summary exit code: $EXIT_CODE"
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
