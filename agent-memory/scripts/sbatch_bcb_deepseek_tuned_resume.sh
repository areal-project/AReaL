#!/bin/bash
#SBATCH --job-name=yl-bcb-ds-tuned-resume
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:5
#SBATCH --exclude=slurmd-24,slurmd-14,slurmd-23,slurmd-45,slurmd-7
#SBATCH --output=logs/bcb_ds_tuned_resume_%j.log
#SBATCH --error=logs/bcb_ds_tuned_resume_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=8000
EMBED_PORT=8001

# Resume from epoch 6 step 750
RESUME_SNAPSHOT="results/deepseek_memrl_tuned/bigcodebench_eval/instruct_full/memory/20260512_183258_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_rl-on/epoch6/snapshot/step_750"
RESUME_EPOCH=6
RESUME_STEP=798

echo "=========================================="
echo "BCB DeepSeek-R1-32B + Qwen3-Embedding-8B (Tuned params - RESUME)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Resume from: epoch ${RESUME_EPOCH} step ${RESUME_STEP}"
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
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start embedding vLLM on GPU 4, port 8001 ---
echo '[INFO] Starting embedding vLLM (Qwen3-Embedding-8B) on GPU 4...'
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
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
    --gpu-memory-utilization 0.90 \
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

# --- Resume MemRL experiment from checkpoint ---
echo ''
echo '=========================================='
echo '[MEMRL-TUNED-RESUME] Resuming BCB MemRL from epoch ${RESUME_EPOCH} step ${RESUME_STEP}'
echo 'Target: complete epoch 6 val + epochs 7-10'
echo '=========================================='

python run/run_bcb.py \
    --config configs/rl_bcb_config.deepseek_tuned.yaml \
    --split instruct \
    --subset full \
    --epochs 10 \
    --checkpoint_interval 50 \
    --max_checkpoints 3 \
    --eval_timeout 240 \
    --untrusted_hard_timeout 300 \
    --strip_think \
    --output_dir ./results/deepseek_memrl_tuned \
    --resume_from ${RESUME_SNAPSHOT} \
    --resume_epoch ${RESUME_EPOCH} \
    --resume_step ${RESUME_STEP}

MEMRL_EXIT=\$?
echo \"[INFO] MemRL-Tuned-Resume exited with code: \$MEMRL_EXIT\"

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
