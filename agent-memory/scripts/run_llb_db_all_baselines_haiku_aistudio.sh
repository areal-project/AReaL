#!/bin/bash
# LLB DB All Baselines — single GPU job, sequential execution with stagger
# Order: selfrag → memp → mem0 → rag → pass@10 (all 10 sections, claude-haiku-4-5)
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_db_all_baselines_haiku_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "LLB DB All Baselines (claude-haiku-4-5)"
echo "Order: selfrag, memp, mem0, rag, pass@10"
echo "All 10 sections each"
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
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
echo "[INFO] MEMRL_LLM_MODEL=$MEMRL_LLM_MODEL"

# ============================================================
# 1. Self-RAG
# ============================================================
echo ""
echo "#################### BASELINE: self-rag ####################"
echo "Start: $(date)"
export MEMRL_RUN_ID=selfrag-haiku-20260714
python3 run/run_llb.py --config configs/rl_llb_db_selfrag.yaml --output_dir $OUTPUT_DIR --self_rag || echo "[WARN] self-rag exited with error $?"
echo "End self-rag: $(date)"
sleep 60

# ============================================================
# 2. MeMp
# ============================================================
echo ""
echo "#################### BASELINE: memp ####################"
echo "Start: $(date)"
export MEMRL_RUN_ID=memp-haiku-20260714
python3 run/run_llb.py --config configs/rl_llb_db_memp.yaml --output_dir $OUTPUT_DIR || echo "[WARN] memp exited with error $?"
echo "End memp: $(date)"
sleep 60

# ============================================================
# 3. Mem0
# ============================================================
echo ""
echo "#################### BASELINE: mem0 ####################"
echo "Start: $(date)"
export MEMRL_RUN_ID=mem0-haiku-20260714
python3 run/run_llb.py --config configs/rl_llb_db_mem0.yaml --output_dir $OUTPUT_DIR --mem0 || echo "[WARN] mem0 exited with error $?"
echo "End mem0: $(date)"
sleep 60

# ============================================================
# 4. RAG
# ============================================================
echo ""
echo "#################### BASELINE: rag ####################"
echo "Start: $(date)"
export MEMRL_RUN_ID=rag-haiku-20260714
python3 run/run_llb.py --config configs/rl_llb_db_rag.yaml --output_dir $OUTPUT_DIR || echo "[WARN] rag exited with error $?"
echo "End rag: $(date)"
sleep 60

# ============================================================
# 5. Pass@10
# ============================================================
echo ""
echo "#################### BASELINE: pass@10 ####################"
echo "Start: $(date)"
export MEMRL_RUN_ID=passk-haiku-20260714
python3 run/run_llb.py --config configs/rl_llb_db_passk.yaml --output_dir $OUTPUT_DIR --baseline_mode passk --baseline_k 10 || echo "[WARN] pass@10 exited with error $?"
echo "End pass@10: $(date)"

echo "=========================================="
echo "All baselines complete!"
echo "End time: $(date)"
echo "=========================================="
