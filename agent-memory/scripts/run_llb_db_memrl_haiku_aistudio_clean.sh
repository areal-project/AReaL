#!/bin/bash
# LLB DB MemRL clean run (paper-aligned) on AIStudio
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_memrl_haiku_clean_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB DB clean run (claude-haiku-4-5)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.5
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing Python dependencies...'
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install -e . failed'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK')"

echo '[INFO] Running LLB DB MemRL clean config...'
python3 run/run_llb.py --config configs/rl_llb_db_memrl_haiku_clean.yaml --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_clean

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
