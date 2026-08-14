#!/bin/bash
# Run LLB OS Self-RAG GPT-4.1-mini on AIStudio only.
# Strict critique mode prevents silent fallback to ordinary RAG.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_selfrag_gpt41mini_${HOST_SHORT}_${TS}.log
mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "LLB OS Self-RAG (GPT-4.1-mini, strict self-critique)"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="
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
export MEMRL_SELFRAG_STRICT=1
export MEMRL_SELFRAG_MAX_ATTEMPTS=8
export MEMRL_SELFRAG_RETRY_DELAY=10
export MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
export MATRIX_REQUEST_INTERVAL=1.5
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd "$PROJECT_DIR"

# AIS eviction retry: reuse this submission's stable run id and latest snapshot.
source "$PROJECT_DIR/scripts/llb_os_auto_resume.sh" "llb_os_selfrag_gpt41mini"
echo "nsenter=$(command -v nsenter) unshare=$(command -v unshare)"
echo '[INFO] Installing runtime deps (memrl imported from source)...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/rl_llb_os_selfrag_gpt41mini.yaml').read_text())
assert cfg['experiment']['task'] == 'os'
assert cfg['experiment']['self_rag'] is True
assert cfg['memory']['k_retrieve'] == 5
print('[Self-RAG] PRECHECK OK: task=os self_rag=true retrieve_k=5 strict=1')
PY
echo '[INFO] Starting GPT-4.1-mini Self-RAG with explicit CLI enablement...'
python3 scripts/run_llb_selfrag_strict.py --config configs/rl_llb_os_selfrag_gpt41mini.yaml --self_rag --self_rag_inject_k 5
echo "=========================================="
echo "LLB OS Self-RAG completed: $(date)"
echo "=========================================="
