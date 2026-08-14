#!/bin/bash
#SBATCH --job-name=yl-bcb-old-bs8
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_ds_old_bs8_%j.log
#SBATCH --error=logs/bcb_ds_old_bs8_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "BCB DeepSeek-R1-32B Baseline (bs=8, compact memory)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Generate temp config with dynamic ports
TEMP_CONFIG=/tmp/baseline_config_\$\$.yaml
sed \"s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g\" configs/rl_bcb_config.deepseek_old_bs8.yaml > \$TEMP_CONFIG
echo \"[INFO] Using ports: LLM=${LLM_PORT}, Embed=${EMBED_PORT}\"

# --- Start embedding vLLM on GPU 4 ---
echo '[INFO] Starting embedding vLLM (Qwen3-Embedding-8B) on GPU 4...'
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --trust-remote-code \
    --disable-log-requests \
    --seed 42 \
    &
EMBED_PID=\$!

echo '[INFO] Waiting for embedding vLLM to be ready...'
for i in \$(seq 1 1200); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] Embedding vLLM is ready!'
        break
    fi
    if [ \$i -eq 1200 ]; then
        echo '[ERROR] Embedding vLLM failed to start after 1200s'
        kill \$EMBED_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Start LLM vLLM on GPUs 0-3 (TP=4) ---
echo '[INFO] Starting LLM vLLM (DeepSeek-R1-32B, TP=4) on GPUs 0-3...'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.80 \
    --disable-log-requests \
    --seed 42 \
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '[INFO] Waiting for LLM vLLM server to be ready...'
for i in \$(seq 1 900); do
    if curl -s http://localhost:${LLM_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] LLM vLLM server is ready!'
        break
    fi
    if [ \$i -eq 900 ]; then
        echo '[ERROR] LLM vLLM server failed to start after 900s'
        kill \$EMBED_PID \$LLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- MemRL baseline bs=8 (compact failure memory) ---
echo ''
echo '=========================================='
echo '[MEMRL-BASELINE-BS8] Starting BCB MemRL baseline (10 epochs, bs=8)'
echo 'Config: tau=0.18, proceduralization, avefact, compact failure insight'
echo '=========================================='

python run/run_bcb.py \
    --config \$TEMP_CONFIG \
    --split instruct \
    --subset full \
    --epochs 10 \
    --checkpoint_interval 50 \
    --max_checkpoints 3 \
    --eval_timeout 240 \
    --untrusted_hard_timeout 300 \
    --strip_think \
    --output_dir ./results/deepseek_memrl_old_bs8

MEMRL_EXIT=\$?
echo \"[INFO] MemRL baseline bs8 exited with code: \$MEMRL_EXIT\"

kill \$EMBED_PID \$LLM_PID 2>/dev/null
wait \$EMBED_PID 2>/dev/null
wait \$LLM_PID 2>/dev/null
echo '[INFO] vLLM servers stopped.'
echo \"[INFO] Final exit code: \$MEMRL_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
