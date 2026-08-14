#!/bin/bash
# ============================================================================
# AIStudio 容器内运行脚本：ALFWorld opus-4-7 Region+FS
# ============================================================================
set -e

LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_alf_opus47_region_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "ALFWorld opus-4-7 Region+FS (AIStudio)"
echo "Start: $(date)"
echo "=========================================="

cd /storage/openpsi/users/yl/agent-memory/MemRL

# --- Install deps ---
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -2
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm hdbscan \
    concurrent-log-handler \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
# textworld/alfworld need special handling (may not be on internal pypi)
pip install textworld alfworld --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3 || true

# --- Environment ---
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm/alf_opus47_ais
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_MIN_INTERVAL=1.5
export MEMRL_EMBED_THROTTLE=0.5
export MEMRL_UPDATE_MAX_WORKERS=2

# Pin run_id so platform retry reuses the same exp dir and auto-resumes from latest ckpt
export MEMRL_RUN_ID="20260623-100806"

echo "[INFO] Verifying imports..."
python3 -c "import memos, memrl; print('[OK] memos + memrl')" || { echo "[FATAL] import failed"; exit 1; }

echo "[INFO] Starting ALFWorld opus-4-7 Region+FS (auto-resume from latest ckpt)..."
python3 run/run_alfworld.py \
    --config configs/rl_alf_config.opus47_region.yaml \
    --region \
    --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 \
    --propagation_eta 0.12 \
    --val_lambda_max 0.15 \
    --failure_summary_n_slots 2 \
    --skip_initial_eval

EXIT_CODE=$?
echo "=========================================="
echo "End: $(date) | Exit: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
