#!/bin/bash
# LLB DB MemRL TRACE debug run (small, paper-aligned) on AIStudio
# Goal: JSONL-trace retrieval to verify Q-values influence eval ranking.
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_memrl_haiku_trace_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs ${PROJECT_DIR}/logs/trace
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB DB TRACE run (claude-haiku-4-5)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

# Source-first import: PROJECT_DIR ahead of everything so /storage source is
# authoritative. No `pip install -e . --target` copy -> code edits take effect
# immediately, including on platform retries.
export PYTHONPATH=${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH}
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

echo '[INFO] Installing runtime deps (NOT memrl itself; memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

# Prove which memrl gets imported (source, not a stale copy).
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

echo '[INFO] Running LLB DB MemRL TRACE config...'
python3 run/run_llb.py --config configs/rl_llb_db_memrl_haiku_trace.yaml

echo "=========================================="
echo "End time: $(date)"
echo "Trace JSONL: ${PROJECT_DIR}/logs/trace/llb_db_trace.jsonl"
echo "=========================================="
