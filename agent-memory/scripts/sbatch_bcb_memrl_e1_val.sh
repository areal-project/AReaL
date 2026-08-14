#!/bin/bash
#SBATCH --job-name=yl-memrl-e1-val
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodelist=slurmd-27
#SBATCH --output=logs/bcb_memrl_e1_val_%j.log
#SBATCH --error=logs/bcb_memrl_e1_val_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

# E1 checkpoint path
E1_CKPT="/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_full/memory/20260428_180242_gpt-4o-2024-11-20_rl-on/epoch1/snapshot/1"

echo "=========================================="
echo "BCB MemRL E1 Val-Only (342 tasks)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "E1 Checkpoint: $E1_CKPT"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB MemRL E1 val-only on \$(hostname)'

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

echo '[INFO] Starting BCB MemRL E1 val-only (resume from E1, run val only)...'
python run/run_bcb.py --config configs/rl_bcb_config.local.yaml --split instruct --subset full --epochs 1 --eval_timeout 240 --untrusted_hard_timeout 300 --split_file configs/bigcodebench/splits/val_only_seed42.json --resume_from $E1_CKPT --resume_epoch 1 --resume_step 0
echo '[INFO] BCB MemRL E1 val-only completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
