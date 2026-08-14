#!/bin/bash
# LLB DB Pass@10 baseline on AIStudio — gpt-4.1-mini
# 10 rounds × 361 tasks, skip already-solved. No val eval.
# round=1 gives independent per-task success (all tasks tried fresh).
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_passk_gpt41mini_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "Pass@10 - LLB DB (gpt-4.1-mini)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.5
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_RUN_ID=passk-db-gpt41mini-20260716
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] MEMRL_LLM_MODEL=$MEMRL_LLM_MODEL"
echo "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"

echo '[INFO] Running LLB DB Pass@10 baseline (gpt-4.1-mini)...'
python3 run/run_llb.py \
    --config configs/rl_llb_db_passk.yaml \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines \
    --baseline_mode passk --baseline_k 10

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
