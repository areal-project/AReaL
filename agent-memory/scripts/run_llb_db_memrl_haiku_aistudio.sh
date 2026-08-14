#!/bin/bash
# LLB DB MemRL (claude-haiku-4-5) — aistudio 容器内启动脚本
set -e

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
LOGFILE=${PROJECT_DIR}/logs/llb_db_memrl_haiku_${HOST_SHORT}_$(date +%Y%m%d_%H%M%S).log
mkdir -p ${PROJECT_DIR}/logs

# 所有 stdout/stderr 同时写到 /storage 日志（可从登录节点 tail -f）
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "MemRL - LLB DB Bench (MemRL, claude-haiku-4-5)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

# --- 环境变量 ---
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH}
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
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages

cd ${PROJECT_DIR}

# --- 安装依赖 ---
echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed (may need fallback DB)'

echo '[INFO] Installing Python dependencies...'
pip install -e . --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install -e . failed'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm \
    concurrent-log-handler mysql-connector-python \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip install deps failed'

echo '[INFO] Verifying imports...'
python3 -c "import memos; import memrl; print('imports OK')"

echo "Python: $(which python3)"
echo "mysqld: $(which mysqld || which mariadbd || find /usr/sbin -name mysqld -o -name mariadbd 2>/dev/null | head -1 || echo 'not found')"

# --- 运行实验 ---
echo '[INFO] Running LLB DB MemRL (claude-haiku-4-5, resume from section 3)...'
python3 run/run_llb.py --config configs/rl_llb_db_memrl_haiku.yaml \
    --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
