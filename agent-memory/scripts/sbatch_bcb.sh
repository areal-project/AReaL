#!/bin/bash
#SBATCH --job-name=yl-memrl-bcb
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodelist=slurmd-27
#SBATCH --output=logs/bcb_submit_%j.log
#SBATCH --error=logs/bcb_submit_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

# Resume settings (set these env vars before sbatch to resume)
# e.g.: RESUME_FROM=.../epoch1/snapshot/1 RESUME_EPOCH=1 sbatch scripts/sbatch_bcb.sh
RESUME_FROM="${RESUME_FROM:-}"
RESUME_EPOCH="${RESUME_EPOCH:-}"
RESUME_STEP="${RESUME_STEP:-}"

echo "=========================================="
echo "BCB Experiment (MemRL)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Resume from: ${RESUME_FROM:-none}"
echo "Resume epoch: ${RESUME_EPOCH:-none}"
echo "Resume step: ${RESUME_STEP:-none}"
echo "=========================================="

# Build resume flags
RESUME_FLAGS=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAGS="--resume_from $RESUME_FROM"
    if [ -n "$RESUME_EPOCH" ]; then
        RESUME_FLAGS="$RESUME_FLAGS --resume_epoch $RESUME_EPOCH"
    fi
    if [ -n "$RESUME_STEP" ]; then
        RESUME_FLAGS="$RESUME_FLAGS --resume_step $RESUME_STEP"
    fi
fi

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies (pqdm, tree-sitter for sanitize)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true

echo '[INFO] Installing BCB eval dependencies (faker, django, statsmodels, etc.)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
# Ensure critical eval packages are installed (requirements-eval.txt has version conflicts)
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '[INFO] Starting BCB experiment...'
python run/run_bcb.py --config configs/rl_bcb_config.local.yaml --split instruct --subset full --epochs ${EPOCHS:-10} --checkpoint_interval 50 --max_checkpoints 3 --eval_timeout 240 --untrusted_hard_timeout 300 $RESUME_FLAGS
echo '[INFO] BCB experiment completed!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
