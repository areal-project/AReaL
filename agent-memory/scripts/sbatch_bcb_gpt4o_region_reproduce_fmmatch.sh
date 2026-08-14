#!/bin/bash
#SBATCH --job-name=yl-bcb-gpt4o-regfm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=380G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/bcb_gpt4o_regfm_%j.log
#SBATCH --error=logs/bcb_gpt4o_regfm_%j.log

# BCB Region + fmmatch on gpt-4o-2024-11-20 + text-embedding-3-large (both API).
# Uses REPRODUCE baseline parameters (tau=0.38, sim_norm=0.31/0.10) for fair comparison.
# fmmatch: failure-mode-matched conditional summary via LLM synthesis.

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
CONFIG=configs/rl_bcb_config.gpt4o_region_reproduce.yaml

echo "=========================================="
echo "BCB Region+fmmatch: gpt-4o + reproduce params (tau=0.38, sim_norm=0.31)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_gpt4o_regfm_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; CONFIG="$2"
cd "$MEMRL_DIR"

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . 2>&1 | tail -2
python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo "[FATAL] memrl import failed"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan 2>&1 | tail -2
python -c 'import memos, mem0' || { echo "[FATAL] memos/mem0 import failed"; exit 1; }
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt 2>&1 | tail -2
echo '[INFO] Installing full BCB eval-time deps (avoid No-module-named false fails)...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt 2>&1 | tail -3 || true
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium \
    django scikit-image pyquery geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein \
    natsort librosa Flask-WTF flask-restful 2>&1 | tail -5 || true
echo '[INFO] All dependencies installed.'

echo '=========================================='
echo '[Region+fmmatch] gpt-4o, reproduce params, 10 epochs, fmmatch enabled'
echo '=========================================='
python run/run_bcb_region.py \
    --config "$CONFIG" \
    --split instruct --subset full --epochs 10 \
    --split_file configs/bigcodebench/splits/full_seed42_dlp_clean_ids.json \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --task_cluster_k 0 \
    --region_gating_mode additive --region_utility_mode beta \
    --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 15 \
    --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 \
    --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 \
    --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 \
    --failure_summary_n_slots 1 \
    --failure_summary_fmmatch \
    --failure_summary_fmmatch_topk 5 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/gpt4o_region_reproduce_fmmatch

EXIT_CODE=$?
echo "[INFO] Region+fmmatch exit code: $EXIT_CODE"
exit $EXIT_CODE
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$CONFIG"
RC=$?
rm -f "$INNER_SCRIPT"
echo "=========================================="
echo "End time: $(date) | Exit code: $RC"
echo "=========================================="
exit $RC
