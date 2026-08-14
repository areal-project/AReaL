#!/bin/bash
# LLB DB Baselines — AIStudio unified launcher
# Runs one baseline at a time. Supports: rag, selfrag, mem0, memp, passk, reflection, nomem
#
# Usage (inside aistudio container):
#   BASELINE=rag bash scripts/run_llb_db_baselines_haiku_aistudio.sh
#
# The LLM model is overridden to claude-haiku-4-5 via MEMRL_LLM_MODEL env var.
set -e

BASELINE="${BASELINE:-rag}"
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_${BASELINE}_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "LLB DB Baseline: ${BASELINE} (claude-haiku-4-5)"
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
# Override LLM model to haiku (runner reads this env var)
export MEMRL_LLM_MODEL=claude-haiku-4-5-20251016
export MEMRL_RUN_ID=${BASELINE}-haiku-20260713
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] BASELINE=$BASELINE"
echo "[INFO] MEMRL_LLM_MODEL=$MEMRL_LLM_MODEL"
echo "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"

OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines

case "${BASELINE}" in
    rag)
        echo '[INFO] Running RAG baseline (retrieve top-k, no Q-value, no RL update)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_rag.yaml \
            --output_dir $OUTPUT_DIR \
            --skip_initial_eval
        ;;
    selfrag)
        echo '[INFO] Running Self-RAG baseline (retrieve + LLM critique filter)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_selfrag.yaml \
            --output_dir $OUTPUT_DIR \
            --self_rag \
            --skip_initial_eval
        ;;
    mem0)
        echo '[INFO] Running Mem0 baseline (mem0 library for memory)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_mem0.yaml \
            --output_dir $OUTPUT_DIR \
            --mem0 \
            --skip_initial_eval
        ;;
    memp)
        echo '[INFO] Running MeMp baseline (proceduralization, no RL)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_memp.yaml \
            --output_dir $OUTPUT_DIR \
            --skip_initial_eval
        ;;
    passk)
        echo '[INFO] Running Pass@k baseline (k=10 retries)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_passk.yaml \
            --output_dir $OUTPUT_DIR \
            --baseline_mode passk --baseline_k 10 \
            --skip_initial_eval
        ;;
    reflection)
        echo '[INFO] Running Reflection baseline (retry with self-reflection)...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_reflection.yaml \
            --output_dir $OUTPUT_DIR \
            --baseline_mode reflection --baseline_k 10 \
            --skip_initial_eval
        ;;
    nomem)
        echo '[INFO] Running No-Memory baseline...'
        python3 run/run_llb.py \
            --config configs/rl_llb_db_nomem_haiku.yaml \
            --output_dir $OUTPUT_DIR \
            --skip_initial_eval
        ;;
    *)
        echo "ERROR: Unknown baseline '${BASELINE}'"
        echo "Usage: BASELINE=<rag|selfrag|mem0|memp|passk|reflection|nomem> bash $0"
        exit 1
        ;;
esac

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
