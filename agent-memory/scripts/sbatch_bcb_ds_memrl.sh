#!/bin/bash
#SBATCH --job-name=yl-bcb-ds-memrl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bcb_ds_memrl_%j.log
#SBATCH --error=logs/bcb_ds_memrl_%j.log

# Must run on the same node as the vLLM server job.
# Usage: sbatch --nodelist=slurmd-XX scripts/sbatch_bcb_ds_memrl.sh

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
VLLM_PORT=8000

echo "=========================================="
echo "BCB DeepSeek MemRL (10 epochs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

echo '[INFO] Installing BCB dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Wait for vLLM server
echo '[INFO] Waiting for vLLM server on localhost:${VLLM_PORT}...'
for i in \$(seq 1 600); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] vLLM server is ready!'
        break
    fi
    if [ \$i -eq 600 ]; then
        echo '[ERROR] vLLM server not reachable after 600s'
        exit 1
    fi
    sleep 1
done

echo '[INFO] Starting BCB MemRL (10 epochs)...'
python run/run_bcb.py \
    --config configs/rl_bcb_config.deepseek.yaml \
    --split instruct \
    --subset full \
    --epochs 10 \
    --checkpoint_interval 50 \
    --max_checkpoints 3 \
    --eval_timeout 240 \
    --untrusted_hard_timeout 300 \
    --strip_think
echo '[INFO] MemRL completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
