#!/bin/bash
# LLB OS baselines run SERIALLY in ONE AIStudio job (one GPU): MemP then RAG.
# Each is API-only (no real GPU compute), so serializing them shares one GPU slot.
# Sandbox default-ON protects /storage. Each uses its own user_id/exp (independent).
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_baselines_memp_rag_gpt41mini_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "LLB OS baselines (serial, one GPU): MemP -> RAG"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
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
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

run_one () {
  local cfg="$1"; local tag; tag=$(basename "$cfg" .yaml)
  echo "=========================================="
  echo "[$(date)] START baseline: $tag"
  echo "=========================================="
  python3 run/run_llb.py --config "$cfg"
  echo "[$(date)] END baseline: $tag (exit=$?)"
}

# Serial order: MemP -> RAG (each full 10-section, with validation after every section).
run_one configs/rl_llb_os_memp_gpt41mini.yaml
run_one configs/rl_llb_os_rag_gpt41mini.yaml

echo "=========================================="
echo "All done (memp + rag): $(date)"
echo "=========================================="
