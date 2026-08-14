#!/bin/bash
#SBATCH --job-name=yl-bcb-holdout-post-b8
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --exclude=slurmd-24
#SBATCH --output=logs/bcb_ds_holdout_post_b8_4gpu_%j.log
#SBATCH --error=logs/bcb_ds_holdout_post_b8_4gpu_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=8002
EMBED_PORT=8003

echo "=========================================="
echo "BCB Holdout Transfer (Post, bs=8, GPU 4-7)"
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

# --- Start embedding vLLM on GPU 4 (single), port 8003 ---
echo '[INFO] Starting embedding vLLM (Qwen3-Embedding-8B) on GPU 4...'
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
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

echo '[INFO] Waiting for embedding vLLM to be ready...'
for i in \$(seq 1 1200); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] Embedding vLLM is ready!'
        break
    fi
    if [ \$i -eq 600 ]; then
        echo '[ERROR] Embedding vLLM failed to start after 1200s'
        kill \$EMBED_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Wait for GPU 0-3 LLM (port 8000) to finish initializing before starting GPU 4-7 LLM
# Two 32B models loading simultaneously exhausts host RAM / /dev/shm
echo '[INFO] Waiting for GPU 0-3 LLM (port 8000) to be ready...'
for i in \$(seq 1 1200); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo '[INFO] GPU 0-3 LLM is ready! Starting GPU 4-7 LLM now.'
        break
    fi
    if [ \$i -eq 1200 ]; then
        echo '[WARN] GPU 0-3 LLM not ready after 1200s, starting anyway.'
    fi
    sleep 1
done

# --- Start LLM vLLM on GPUs 4-7 (TP=4), port 8002 ---
echo '[INFO] Starting LLM vLLM (DeepSeek-R1-32B, TP=4) on GPUs 4-7...'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.70 \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!
echo \"[INFO] LLM vLLM PID: \$LLM_PID\"

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

echo ''
echo '=========================================='
echo '[HOLDOUT-POST-B8] Cross-Subtask Transfer Experiment'
echo 'Config: old (tau=0.18, trajectory, avefact), bs=8'
echo 'Region: Post retrieval, multiplicative gating'
echo 'Holdout: all 7 subtasks, 8 banks per task'
echo '=========================================='

python run/run_bcb_holdout.py \
    --config configs/rl_bcb_config.deepseek_old_holdout_bs8_gpu47.yaml \
    --split instruct \
    --subset full \
    --epochs 5 \
    --checkpoint_interval 50 \
    --max_checkpoints 3 \
    --eval_timeout 240 \
    --untrusted_hard_timeout 300 \
    --strip_think \
    --region_gating_mode multiplicative \
    --region_utility_mode beta \
    --region_temperature 1.0 \
    --resume_from ./results/deepseek_old_holdout_post_b8/bigcodebench_eval/instruct_full/holdout/20260519_144040_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_holdout/epoch1/snapshot/step_352 \
    --resume_epoch 1 \
    --resume_step 352 \
    --output_dir ./results/deepseek_old_holdout_post_b8

HOLDOUT_EXIT=\$?
echo \"[INFO] Holdout-Post-B8 exited with code: \$HOLDOUT_EXIT\"

kill \$EMBED_PID \$LLM_PID 2>/dev/null
wait \$EMBED_PID 2>/dev/null
wait \$LLM_PID 2>/dev/null
echo '[INFO] vLLM servers stopped.'
echo \"[INFO] Final exit code: \$HOLDOUT_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
