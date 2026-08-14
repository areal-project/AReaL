#!/bin/bash
# LLB DB MemRL run with CORRECTED reflection prompt (v2) on AIStudio
# v2 stops the reflection from misdiagnosing "output format mismatch" and teaches
# the evaluator's real tuple/Decimal/order contract. Answer-preserving truncation
# keeps the submitted `Final Answer:` in the reflection window.
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_memrl_haiku_v2reflect_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB DB v2-reflection run (claude-haiku-4-5)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

# Source-first import: PROJECT_DIR ahead of everything; no `pip install -e . --target`
# copy, so code edits (v2 reflection) take effect immediately, incl. platform retries.
export PYTHONPATH=${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.5
# >>> the switch: use the corrected reflection prompt <<<
export MEMRL_LLB_REFLECTION_PROMPT=v2
# >>> DB-specific: generate concise SQL pattern memories instead of generic templates <<<
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
# Pin the ckpt dir id so a platform retry (preemption) writes to the SAME
# exp_<name>_<run_id>/ dir -> auto-resume picks up the latest section/batch snapshot
# instead of starting a fresh run. MUST be a fixed constant (not date-generated).
export MEMRL_RUN_ID=v2reflect-dbpattern-20260712
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
echo "[INFO] MEMRL_LLB_REFLECTION_PROMPT=$MEMRL_LLB_REFLECTION_PROMPT"

echo '[INFO] Running LLB DB MemRL v2-reflection config...'
python3 run/run_llb.py --config configs/rl_llb_db_memrl_haiku_v2reflect.yaml --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
