#!/bin/bash
# LLB OS baselines run SERIALLY in ONE AIStudio job (one GPU): MemP then RAG.
# Each is API-only (no real GPU compute), so serializing them shares one GPU slot.
# Sandbox default-ON protects /storage. Each uses its own user_id/exp (independent).
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_baselines_memp_rag_${HOST_SHORT}_${TS}.log
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
export MEMRL_EMBED_THROTTLE=1.0
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

# Serial order: val eval (fast, inference-only) -> MemP -> RAG (each full 10-section).
echo "=========================================="
echo "[$(date)] START: MemRL val eval from checkpoints (inference only, ~few hours)"
echo "=========================================="
python3 scripts/eval_memrl_val_from_ckpts.py
echo "[$(date)] END: MemRL val eval (exit=$?)"

run_one configs/rl_llb_os_memp_haiku.yaml
run_one configs/rl_llb_os_rag_haiku.yaml

echo "=========================================="
echo "All done (val eval + memp + rag): $(date)"
echo "=========================================="
