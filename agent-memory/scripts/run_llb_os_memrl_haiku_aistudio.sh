#!/bin/bash
# LLB OS MemRL on AIStudio with claude-haiku-4-5.
# Uses MEMRL_OS_SANDBOX=1 so agent bash runs in an unshare mount-namespace that
# tmpfs-covers /storage (zero risk to real /storage) + /etc, plus L1 interception
# of any command referencing /storage. Validates v2 OS-reflection + detailed script.
set +e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_memrl_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB OS MemRL (claude-haiku-4-5, sandboxed)"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
# >>> the safety switch: sandbox agent bash away from /storage <<<
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
# Bumped 0.5 -> 1.0 to reduce embedding request rate (matrixllm 429 rate limits
# hit the earlier run). Higher = slower but fewer 429 retries.
export MEMRL_EMBED_THROTTLE=1.0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] MEMRL_OS_SANDBOX=$MEMRL_OS_SANDBOX"

echo '[INFO] Running LLB OS MemRL (haiku, sandboxed)...'
python3 run/run_llb.py --config configs/rl_llb_os_memrl_haiku.yaml

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
