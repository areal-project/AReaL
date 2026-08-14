#!/bin/bash
# LLB OS MemRL + Region + Failure-Summary (claude-haiku-4-5) on AIStudio.
# Region/FS enabled via CLI flags below (config yaml stays plain MemRL). Sandbox
# default-ON protects /storage. Independent user_id/exp dir from the MemRL run.
# HOLD: submit only when the MemRL run is nearly done (avoid API contention).
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_region_fs_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB OS Region+FS (claude-haiku-4-5, sandboxed)"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
# Widen throttles: this runs alongside MemRL + pass@10, so space out API sends to
# avoid matrixllm 429 contention (3 jobs sharing quota). Slower but fewer retries.
export MEMRL_EMBED_THROTTLE=1.5
export MEMRL_LLM_MIN_INTERVAL=1.0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler hdbscan --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

echo '[INFO] Running LLB OS MemRL Region+FS (haiku, sandboxed)...'
python3 run/run_llb.py --config configs/rl_llb_os_region_haiku.yaml \
  --region --region_k 8 --region_gating_mode additive \
  --failure_summary_n_slots 2 --failure_summary_k 10 \
  --explore_schedule 0,2,2,1,1,1,1,0,0,0

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
