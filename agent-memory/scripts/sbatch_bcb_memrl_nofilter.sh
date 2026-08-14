#!/bin/bash
#SBATCH --job-name=yl-memrl-nofilter
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/bcb_memrl_nofilter_%j.log
#SBATCH --error=logs/bcb_memrl_nofilter_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "BCB MemRL NoFilter (git original config, 750+300 clean tasks)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB MemRL NoFilter on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true

echo '[INFO] Installing BCB eval dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '[INFO] Starting BCB MemRL NoFilter (git original config, 10 epochs, 750 train + 300 val)...'
python run/run_bcb.py --config configs/rl_bcb_config.git_original.yaml --split instruct --subset full --epochs 10 --eval_timeout 240 --untrusted_hard_timeout 300 --checkpoint_interval 50 --max_checkpoints 3 --split_file configs/bigcodebench/splits/full_seed42_nofilter.json
echo '[INFO] BCB MemRL NoFilter completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
