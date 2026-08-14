#!/bin/bash
#SBATCH --job-name=yl-memrl-resume-e6
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/bcb_memrl_resume_e6_%j.log
#SBATCH --error=logs/bcb_memrl_resume_e6_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
E6_CKPT="/storage/openpsi/users/yl/agent-memory/MemRL/results/bigcodebench_eval/instruct_full/memory/20260429_223057_gpt-4o-2024-11-20_rl-on/epoch6/snapshot/step_350"

echo "=========================================="
echo "BCB MemRL Resume from E6 step 350 (local config, 128G mem)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "E6 Checkpoint: $E6_CKPT"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB MemRL resume from E6 step 350 on \$(hostname)'

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

echo '[INFO] Starting BCB MemRL resume from E6 step 350 (E6-E10, local config, 128G mem)...'
python run/run_bcb.py --config configs/rl_bcb_config.local.yaml --split instruct --subset full --epochs 10 --eval_timeout 240 --untrusted_hard_timeout 300 --checkpoint_interval 50 --max_checkpoints 3 --resume_from $E6_CKPT --resume_epoch 6 --resume_step 350
echo '[INFO] BCB MemRL resume from E6 completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
