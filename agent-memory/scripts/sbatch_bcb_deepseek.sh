#!/bin/bash
#SBATCH --job-name=yl-bcb-deepseek
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24
#SBATCH --output=logs/bcb_deepseek_%j.log
#SBATCH --error=logs/bcb_deepseek_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
VLLM_PORT=8000

echo "=========================================="
echo "BCB DeepSeek-R1-Distill-Qwen-32B (nomem + memrl PARALLEL)"
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
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start vLLM server in background ---
echo '[INFO] Starting vLLM server with DeepSeek-R1-Distill-Qwen-32B (TP=8)...'
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 8 \
    --port ${VLLM_PORT} \
    --trust-remote-code \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    &
VLLM_PID=\$!
echo \"[INFO] vLLM PID: \$VLLM_PID\"

# Set offline mode for experiment scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Wait for vLLM to be ready
echo '[INFO] Waiting for vLLM server to be ready...'
for i in \$(seq 1 600); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] vLLM server is ready!'
        break
    fi
    if [ \$i -eq 600 ]; then
        echo '[ERROR] vLLM server failed to start after 600s'
        kill \$VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Run BOTH experiments in parallel ---
echo ''
echo '=========================================='
echo '[PARALLEL] Starting No-Memory + MemRL concurrently'
echo '=========================================='

# No-Memory Baseline (background)
(
    echo '[NOMEM] Starting BCB No-Memory baseline (1 epoch, retrieve_k=0)...'
    python run/run_bcb.py \
        --config configs/rl_bcb_config.deepseek.yaml \
        --split instruct \
        --subset full \
        --epochs 1 \
        --retrieve_k 0 \
        --checkpoint_interval 50 \
        --max_checkpoints 3 \
        --eval_timeout 240 \
        --untrusted_hard_timeout 300 \
        --strip_think \
        --output_dir ./results/deepseek_nomem
    echo '[NOMEM] No-Memory baseline completed!'
) > logs/bcb_ds_nomem_\${SLURM_JOB_ID}.log 2>&1 &
NOMEM_PID=\$!

# MemRL (background)
(
    echo '[MEMRL] Starting BCB MemRL (10 epochs)...'
    python run/run_bcb.py \
        --config configs/rl_bcb_config.deepseek.yaml \
        --split instruct \
        --subset full \
        --epochs 10 \
        --checkpoint_interval 50 \
        --max_checkpoints 3 \
        --eval_timeout 240 \
        --untrusted_hard_timeout 300 \
        --strip_think \
        --output_dir ./results/deepseek_memrl
    echo '[MEMRL] MemRL completed!'
) > logs/bcb_ds_memrl_\${SLURM_JOB_ID}.log 2>&1 &
MEMRL_PID=\$!

echo \"[INFO] NOMEM PID: \$NOMEM_PID, MEMRL PID: \$MEMRL_PID\"

# Wait for both to finish
wait \$NOMEM_PID
NOMEM_EXIT=\$?
echo \"[INFO] NOMEM exited with code: \$NOMEM_EXIT\"

wait \$MEMRL_PID
MEMRL_EXIT=\$?
echo \"[INFO] MEMRL exited with code: \$MEMRL_EXIT\"

# Cleanup vLLM
kill \$VLLM_PID 2>/dev/null
wait \$VLLM_PID 2>/dev/null
echo '[INFO] vLLM server stopped.'
echo \"[INFO] Final exit codes: NOMEM=\$NOMEM_EXIT, MEMRL=\$MEMRL_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
