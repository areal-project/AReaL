#!/bin/bash
#SBATCH --job-name=yl-bcb-reeval-pkg
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/bcb_reeval_pkg_%j.log
#SBATCH --error=logs/bcb_reeval_pkg_%j.log

# Re-evaluate ONLY the no-mem tasks that failed with "No module named X" — install the
# full BCB eval dependency set first, then re-run their stored solutions through the
# official untrusted_check. CPU-only; the model outputs already exist in samples.jsonl.

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
RUN_DIR="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_local_nomem/bigcodebench_eval/instruct_full/memory/20260618_173854_deepseek-ai_DeepSeek-V3_rl-on"

echo "=========================================="
echo "BCB re-eval of missing-package no-mem tasks"
echo "Run dir: $RUN_DIR"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing memrl + BCB eval deps...'
pip install -e . --quiet 2>&1 | tail -2
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>&1 | tail -2
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -3 || true
# Full eval dependency set (the packages whose absence caused the false failures).
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium \
    django scikit-image pyquery geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein \
    natsort librosa Flask-WTF flask-restful tensorflow keras --quiet 2>&1 | tail -5 || true
echo '[INFO] Running re-eval...'
python scripts/reeval_missing_pkg_tasks.py --run_dir '${RUN_DIR}' --subset full
"
RC=$?
echo "=========================================="
echo "End time: $(date) | Exit: $RC"
echo "=========================================="
exit $RC
