#!/bin/bash
#SBATCH --job-name=yl-bcb-dsv3-api
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/bcb_dsv3_api_%j.log
#SBATCH --error=logs/bcb_dsv3_api_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "BCB DeepSeek-V3.1 (API): NoMem → MemRL baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "No GPU needed — LLM + Embedding via MatrixLLM API"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/bcb_dsv3_api_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"
cd "$MEMRL_DIR"

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . 2>&1 | tail -3
python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo "[FATAL] memrl import failed"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan 2>&1 | tail -3
python -c 'import memos, mem0' || { echo "[FATAL] memos/mem0 import failed"; exit 1; }
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt 2>&1 | tail -3
echo '[INFO] Installing critical eval-time modules...'
pip install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium 2>&1 | tail -5
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt 2>&1 | tail -5 || true
echo '[INFO] All dependencies installed.'

CONFIG=configs/rl_bcb_config.deepseek_v3.yaml

# ==== Phase 1: No-Memory baseline (1 epoch, retrieve_k=0) ====
echo '=========================================='
echo '[Phase 1] No-Memory baseline (retrieve_k=0, 1 epoch)'
echo '=========================================='
python run/run_bcb.py \
    --config "$CONFIG" \
    --split instruct --subset full --epochs 1 \
    --retrieve_k 0 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_api_nomem
echo "[INFO] No-Memory exit code: $?"

# ==== Phase 2: MemRL baseline (3 epochs, retrieve_k=10) ====
echo '=========================================='
echo '[Phase 2] MemRL baseline (retrieve_k=10, 3 epochs)'
echo '=========================================='
python run/run_bcb.py \
    --config "$CONFIG" \
    --split instruct --subset full --epochs 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/deepseek_v3_api_memrl

EXIT_CODE=$?
echo "[INFO] MemRL exit code: $EXIT_CODE"
echo '[INFO] Done.'
exit $EXIT_CODE
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR"
RC=$?
rm -f "$INNER_SCRIPT"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $RC"
echo "=========================================="
exit $RC
