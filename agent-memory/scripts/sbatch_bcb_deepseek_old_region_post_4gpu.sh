#!/bin/bash
#SBATCH --job-name=yl-bcb-old-post-4gpu
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --exclude=slurmd-24
#SBATCH --output=logs/bcb_ds_old_region_post_4gpu_%j.log
#SBATCH --error=logs/bcb_ds_old_region_post_4gpu_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=8000
EMBED_PORT=8001

echo "=========================================="
echo "BCB DeepSeek-R1-32B + Qwen3-Embedding-8B (Old Config + Region Post, 4 GPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPU Config: Embedding TP=4 + LLM TP=4 on GPUs 0-3"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting on \$(hostname)'

echo '[INFO] Switching to region-dev branch...'
git checkout region-dev --quiet 2>/dev/null || true
echo \"[INFO] Current branch: \$(git branch --show-current)\"

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

# --- Start embedding vLLM on GPU 0 (single), port 8001 ---
echo '[INFO] Starting embedding vLLM (Qwen3-Embedding-8B) on GPU 0...'
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code \
    --disable-log-requests \
    &
EMBED_PID=\$!
echo \"[INFO] Embedding vLLM PID: \$EMBED_PID\"

# Wait for embedding server
echo '[INFO] Waiting for embedding vLLM to be ready...'
for i in \$(seq 1 600); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] Embedding vLLM is ready!'
        break
    fi
    if [ \$i -eq 600 ]; then
        echo '[ERROR] Embedding vLLM failed to start after 600s'
        kill \$EMBED_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Start LLM vLLM on GPUs 0-3 (TP=4), port 8000 ---
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
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!
echo \"[INFO] LLM vLLM PID: \$LLM_PID\"

# Set offline mode for experiment scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Wait for LLM vLLM to be ready
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

# --- Run MemRL experiment with old config + region post (bs=1) ---
echo ''
echo '=========================================='
echo '[MEMRL-OLD-REGION-POST] Starting BCB MemRL (10 epochs, bs=1)'
echo 'Config: tau=0.18, trajectory, avefact, sim_norm=0.186/0.094'
echo 'Region: Post retrieval, multiplicative gating (default)'
echo '=========================================='

python run/run_bcb_region.py \
    --config configs/rl_bcb_config.deepseek_old_region_post_bs1.yaml \
    --split instruct \
    --subset full \
    --epochs 10 \
    --checkpoint_interval 50 \
    --max_checkpoints 3 \
    --eval_timeout 240 \
    --untrusted_hard_timeout 300 \
    --strip_think \
    --region_retrieve_mode post \
    --region_gating_mode multiplicative \
    --output_dir ./results/deepseek_old_region_post_4gpu

MEMRL_EXIT=\$?
echo \"[INFO] MemRL-Old-Region-Post exited with code: \$MEMRL_EXIT\"

# Cleanup vLLM servers
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
