#!/bin/bash
#SBATCH --job-name=yl-bcb-dsv3-memrl-fs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/bcb_dsv3_memrl_fs_%j.log
#SBATCH --error=logs/bcb_dsv3_memrl_fs_%j.log

# BCB DeepSeek-V3 (original instruct, 671B FP8) fully local.
# MemRL + Failure Summary (inline aggregation, no region clustering).
# - LLM: TP8 + Expert Parallel on all 8 H200
# - Embedding: Qwen3-Embedding-8B co-located on GPU 0

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-dev-vllm-20260429.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/users/yl/models/DeepSeek-V3-mtp1"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=8000
EMBED_PORT=8001
CONFIG=configs/rl_bcb_config.deepseek_v3_local.yaml

echo "=========================================="
echo "BCB DeepSeek-V3 MemRL + Failure Summary (inline)"
echo "vLLM image:   $VLLM_IMG"
echo "Runner image: $RUNNER_IMG"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# Free ports from any stale server holding them.
fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
fuser -k ${EMBED_PORT}/tcp 2>/dev/null || true
sleep 5

# ============================================================
# 1) Embedding server (Qwen3-Embedding-8B) pinned to GPU 0.
# ============================================================
echo "[INFO] Launching embedding vLLM (Qwen3-Embedding-8B) on GPU 0 (image: $VLLM_IMG)..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $VLLM_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.12 \
    --trust-remote-code
" &
EMBED_PID=$!
echo "[INFO] Embedding vLLM PID: $EMBED_PID"

echo "[INFO] Waiting for embedding vLLM to be ready..."
for i in $(seq 1 600); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then
        echo "[INFO] Embedding vLLM is ready!"
        break
    fi
    if ! kill -0 $EMBED_PID 2>/dev/null; then
        echo "[ERROR] Embedding vLLM process died during startup."
        exit 1
    fi
    if [ $i -eq 600 ]; then
        echo "[ERROR] Embedding vLLM failed to start after 600s"
        kill $EMBED_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ============================================================
# 2) LLM server: DeepSeek-V3 TP8 + Expert Parallel across all 8 GPUs.
# ============================================================
echo "[INFO] Launching LLM vLLM (DeepSeek-V3, TP8+EP) on all 8 GPUs..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $VLLM_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.78
" &
VLLM_PID=$!
echo "[INFO] LLM vLLM PID: $VLLM_PID"

echo "[INFO] Waiting for LLM vLLM (real chat probe, up to 90min for FP8 startup)..."
for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-ai/DeepSeek-V3","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$PROBE" = "200" ]; then
        echo "[INFO] LLM vLLM is ready (chat probe 200)!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] LLM vLLM process died during startup."
        kill $EMBED_PID 2>/dev/null
        exit 1
    fi
    if [ $i -eq 5400 ]; then
        echo "[ERROR] LLM vLLM failed to start after 5400s (last probe: $PROBE)"
        kill $EMBED_PID $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ============================================================
# 3) Run the experiment in the RUNNER image (mem packages).
# ============================================================
run_in_runner() {
    singularity exec --nv --no-home --writable-tmpfs \
        --bind /storage:/storage \
        $RUNNER_IMG \
        bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing runner dependencies...'
find . -name '*.pyc' -delete 2>/dev/null || true
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . 2>&1 | tail -2
python -c 'import memrl; print(\"[OK] memrl:\", memrl.__file__)' || { echo '[FATAL] memrl import failed'; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>&1 | tail -2
python -c 'import memos, mem0' || { echo '[FATAL] memos/mem0 import failed'; exit 1; }
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>&1 | tail -2
echo '[INFO] Installing critical eval-time modules...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -3 || true
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium \
    django scikit-image pyquery geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
\$1
"
}

# ==== MemRL + Failure Summary (3 epochs, retrieve_k=10, failure_summary_n_slots=1) ====
echo ''
echo '=========================================='
echo '[MemRL+FS] DeepSeek-V3, 3 epochs, failure_summary_n_slots=1 (inline)'
echo '=========================================='
run_in_runner "python run/run_bcb.py \
    --config $CONFIG \
    --split instruct --subset full --epochs 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --failure_summary_n_slots 1 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_memrl_fs"
MEMRL_FS_EXIT=$?
echo "[INFO] MemRL+FS exit code: $MEMRL_FS_EXIT"

# Cleanup servers.
kill $EMBED_PID $VLLM_PID 2>/dev/null
wait $EMBED_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[INFO] Servers stopped. MemRL+FS exit: $MEMRL_FS_EXIT"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
