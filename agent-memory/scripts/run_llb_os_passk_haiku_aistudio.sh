#!/bin/bash
# LLB OS pass@10 baseline (claude-haiku-4-5) on AIStudio.
# round1 = nomem baseline, round10 = intrinsic ceiling. Sandbox default-ON.
# HOLD: only submit after seeing whether MemRL improves.
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_passk_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "LLB OS pass@10 baseline (claude-haiku-4-5, sandboxed)"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
# sandbox default-ON already; set explicitly for clarity
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=1.0
# Stagger against the running MemRL job: widen per-sample interval in pass@k rounds
# (default 0.5s) so the two jobs don't collide on matrixllm API quota (429s).
export MEMRL_LLB_REQUEST_INTERVAL=2.0
export MEMRL_LLM_MIN_INTERVAL=0.8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

echo '[INFO] Running LLB OS pass@10 baseline (haiku, sandboxed)...'
python3 run/run_llb.py --config configs/rl_llb_os_passk_haiku.yaml

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
