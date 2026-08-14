#!/bin/bash
#SBATCH --job-name=yl-bcb-nomem
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodelist=slurmd-27
#SBATCH --output=logs/bcb_nomem_%j.log
#SBATCH --error=logs/bcb_nomem_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "BCB No-Memory Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB No-Memory baseline on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies (pqdm, tree-sitter for sanitize)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true

echo '[INFO] Installing BCB eval dependencies (faker, django, statsmodels, etc.)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '[INFO] Starting BCB No-Memory baseline (retrieve_k=0, epochs=1)...'
python run/run_bcb.py --config configs/rl_bcb_config.local.yaml --split instruct --subset full --epochs 1 --retrieve_k 0 --eval_timeout 240 --untrusted_hard_timeout 300 --checkpoint_interval 50 --max_checkpoints 3
echo '[INFO] BCB No-Memory baseline completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
