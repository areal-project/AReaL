#!/bin/bash
# LLB DB Region+FS (Our Method) on AIStudio — claude-haiku-4-5
# Region-aware memory with failure summary injection.
# Parameters aligned with ALFWorld/BCB region+FS runs.
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_region_fs_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "Region+FS - LLB DB (claude-haiku-4-5)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.5
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=claude-haiku-4-5-20251016
export MEMRL_RUN_ID=region-fs-db-20260713
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] MEMRL_LLB_REFLECTION_PROMPT=$MEMRL_LLB_REFLECTION_PROMPT"
echo "[INFO] MEMRL_LLB_SCRIPT_DETAIL=$MEMRL_LLB_SCRIPT_DETAIL"
echo "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"

echo '[INFO] Running LLB DB Region+FS...'
python3 run/run_llb.py \
    --config configs/rl_llb_db_region_fs.yaml \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect \
    --region --region_gating_mode additive \
    --region_utility_mode beta \
    --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 15 \
    --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 \
    --propagation_sim_min 0.40 --shrinkage_confidence_k 3.0 \
    --val_lambda_max 0.15 \
    --no_z_norm \
    --explore_schedule "0,4,3,2,2,1,1,1,1,0" \
    --explore_success_ratio 0.7 \
    --failure_summary_n_slots 2 \
    --skip_initial_eval

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
