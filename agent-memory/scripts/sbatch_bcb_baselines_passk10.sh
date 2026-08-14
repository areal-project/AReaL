#!/bin/bash
#SBATCH --job-name=yl-bcb-passk10
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-3
#SBATCH --output=logs/bcb_passk10_%j.log
#SBATCH --error=logs/bcb_passk10_%j.log

# BCB pass@10 baseline: 10 independent attempts per task, no memory
# Uses --baseline_mode passk for proper cumulative SR tracking

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SGLANG_IMG="/storage/openpsi/images/sglang-v0.5.10.sif"
VLLM_IMG="/storage/openpsi/images/areal-dev-vllm-20260429.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/users/yl/models/DeepSeek-V3-mtp1"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=8000
EMBED_PORT=8001
CONFIG=configs/rl_bcb_config.passk10_local.yaml

echo "=========================================="
echo "BCB Pass@10 Baseline: DeepSeek-V3 via SGLang"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
fuser -k ${EMBED_PORT}/tcp 2>/dev/null || true
sleep 5

echo "[INFO] Launching SGLang server (DeepSeek-V3, TP=8)..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SGLANG_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --trust-remote-code \
    --host 127.0.0.1 --port ${LLM_PORT} \
    --context-length 32768 \
    --mem-fraction-static 0.70
" &
VLLM_PID=$!

echo "[INFO] Waiting for SGLang server..."
for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-ai/DeepSeek-V3","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$PROBE" = "200" ]; then echo "[INFO] SGLang ready!"; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[ERROR] SGLang died"; exit 1; fi
    if [ $i -eq 5400 ]; then echo "[ERROR] SGLang timeout"; kill $VLLM_PID 2>/dev/null; exit 1; fi
    sleep 1
done

echo "[INFO] Launching embedding vLLM (Qwen3-Embedding-8B) on GPU 0..."
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

echo "[INFO] Waiting for embedding vLLM..."
for i in $(seq 1 600); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then echo "[INFO] Embedding ready!"; break; fi
    if ! kill -0 $EMBED_PID 2>/dev/null; then echo "[ERROR] Embedding died"; kill $VLLM_PID 2>/dev/null; exit 1; fi
    if [ $i -eq 600 ]; then echo "[ERROR] Embedding timeout"; kill $EMBED_PID $VLLM_PID 2>/dev/null; exit 1; fi
    sleep 1
done

run_in_runner() {
    singularity exec --nv --no-home --writable-tmpfs \
        --bind /storage:/storage \
        $RUNNER_IMG \
        bash -c "
cd ${MEMRL_DIR}
find . -name '*.pyc' -delete 2>/dev/null || true
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . 2>&1 | tail -2
python -c 'import memrl; print(\"[OK] memrl:\", memrl.__file__)' || { echo '[FATAL] memrl import failed'; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>&1 | tail -2
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>&1 | tail -2
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

echo ''
echo '=========================================='
echo '[pass@10] DeepSeek-V3, baseline_mode=passk, k=10'
echo '=========================================='
run_in_runner "python run/run_bcb.py \
    --config $CONFIG \
    --split instruct --subset full \
    --baseline_mode passk --baseline_k 10 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_passk10_baseline"
EXIT_CODE=$?
echo "[INFO] pass@10 exit code: $EXIT_CODE"

kill $EMBED_PID $VLLM_PID 2>/dev/null
wait $EMBED_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[INFO] Done. Exit: $EXIT_CODE | End: $(date)"
