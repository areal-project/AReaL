#!/bin/bash
# Resume LLB OS MemP GPT-4.1-mini from the original run's snapshot/7.
# First replays interrupted S7 validation exactly once, then trains/evals S8-S10.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_memp_gpt41mini_resume_s7_${HOST_SHORT}_${TS}.log
mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "$LOGFILE") 2>&1

# Critical: reuse the original ck_dir. AIS retries auto-load its newest valid snapshot.
export MEMRL_RUN_ID=20260718-035420
export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH:-}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=1.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=1.0
export MEMRL_LLB_REQUEST_INTERVAL=1.5
export MEMRL_EMBED_MAX_RETRIES=8
export MEMRL_EMBED_429_BASE_DELAY=10
export MEMRL_EMBED_429_MAX_DELAY=120
export MEMRL_EMBED_RETRY_JITTER=2
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.cache/embedding_rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=llb-os-text-embedding-3-large
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd "$PROJECT_DIR"
echo "LLB OS MemP resume: S7 validation -> S8-S10; start=$(date); log=$LOGFILE"
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
python3 run/run_llb.py \
  --config configs/rl_llb_os_memp_gpt41mini_resume_s7.yaml \
  --resume_eval_section -1

echo "LLB OS MemP resume completed: $(date)"
