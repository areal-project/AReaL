#!/bin/bash
# RESUME region+FS from snapshot/1_b69 (Section 1 completed, continue S2-S10).
# Now with valid_interval=1 so val SR is computed every epoch.
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_region_fs_gpt41mini_resume_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "RESUME LLB OS Region+FS (claude-gpt-4.1-mini) from ckpt"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
# Widened further after sustained 429s (job空转37min全429). Space sends out more.
# Shared embedding throttle/backoff for all LLB OS gpt-4.1-mini AIS jobs.
# The shared /storage state + common key coordinates separate AIS containers.
export MEMRL_EMBED_THROTTLE=1.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=1.0
export MEMRL_LLB_REQUEST_INTERVAL=1.5
export MEMRL_EMBED_MAX_RETRIES=8
export MEMRL_EMBED_429_BASE_DELAY=10
export MEMRL_EMBED_429_MAX_DELAY=120
export MEMRL_EMBED_RETRY_JITTER=2
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.cache/embedding_rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=llb-os-text-embedding-3-large
export MEMRL_LLM_MIN_INTERVAL=3.0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
# Pin exp dir of the stalled run -> auto-resumes from snapshot/1_b69 (Section 2 batch 0)
export MEMRL_RUN_ID=20260715-114747

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)  MEMRL_RUN_ID=$MEMRL_RUN_ID"
echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler hdbscan --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'
python3 -c "import memos, memrl; print('imports OK')"

echo '[INFO] Resuming Region+FS from latest snapshot...'
python3 run/run_llb.py --config configs/rl_llb_os_region_gpt41mini.yaml \
  --region --region_k 8 --region_gating_mode additive \
  --failure_summary_n_slots 2 --failure_summary_k 10 \
  --explore_schedule 0,2,2,1,1,1,1,0,0,0

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
