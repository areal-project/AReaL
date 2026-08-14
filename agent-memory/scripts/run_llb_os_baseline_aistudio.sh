#!/bin/bash
# Generic LLB OS baseline runner for AIStudio (RAG / MemP / Mem0 / Self-RAG).
# Usage: run_llb_os_baseline_aistudio.sh <config.yaml> ["<extra run_llb args>"]
#   e.g. ... rl_llb_os_mem0_haiku.yaml "--mem0 --mem0_infer true"
# Falls back to OS_BASELINE_CONFIG/OS_BASELINE_EXTRA_ARGS env if args absent.
# Sandbox default-ON.
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
CFG="${1:-${OS_BASELINE_CONFIG:?set config as arg1 or OS_BASELINE_CONFIG}}"
EXTRA="${2:-${OS_BASELINE_EXTRA_ARGS:-}}"
TAG=$(basename "$CFG" .yaml)
LOGFILE=${PROJECT_DIR}/logs/${TAG}_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "LLB OS baseline: $TAG (sandboxed)"
echo "Host: $(hostname)  Start: $(date)  Config: $CFG  Extra: ${EXTRA:-none}"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=1.0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

echo "[INFO] Running $TAG ..."
python3 run/run_llb.py --config "$CFG" ${EXTRA}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="

