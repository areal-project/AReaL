#!/bin/bash
# LLB DB Region+FS (Our Method) on AIStudio — gpt-4.1-mini
# Region-aware memory with failure summary injection.
# Parameters aligned with ALFWorld/BCB region+FS runs.
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_region_fs_gpt41mini_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "Region+FS - LLB DB (gpt-4.1-mini)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=1.5
export MEMRL_EMBED_MIN_INTERVAL=2.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=1.5
export MEMRL_EMBED_429_BASE_DELAY=5.0
export MEMRL_EMBED_429_MAX_DELAY=60.0
export MEMRL_EMBED_RETRY_JITTER=1.0
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=matrixllm-text-embedding-3-large
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_RUN_ID=region-fs-db-gpt41mini-20260716
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] MEMRL_LLM_MODEL=$MEMRL_LLM_MODEL"
echo "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"

echo '[INFO] Running LLB DB Region+FS (gpt-4.1-mini)...'
python3 run/run_llb.py \
    --config configs/rl_llb_db_region_fs.yaml \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect \
    --region --region_gating_mode additive \
    --propagation_eta 0.12 \
    --shrinkage_confidence_k 3.0 \
    --no_z_norm \
    --explore_schedule "0,4,3,2,2,1,1,1,1,0" \
    --failure_summary_n_slots 2

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
