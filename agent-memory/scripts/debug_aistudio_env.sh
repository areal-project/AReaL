#!/bin/bash
# Debug 脚本：验证 aistudio 容器环境是否适合跑 LLB DB MemRL
set -x

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOGFILE=${PROJECT_DIR}/logs/aistudio_debug_$(date +%Y%m%d_%H%M%S).log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "DEBUG: LLB DB MemRL 环境验证"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# 1. 基本环境
echo "=== [1/7] 系统信息 ==="
uname -a
whoami
id
cat /etc/os-release 2>/dev/null | head -5

# 2. Python 环境
echo "=== [2/7] Python 环境 ==="
which python3
python3 --version
which pip
pip --version
echo "sys.path:"
python3 -c "import sys; print('\n'.join(sys.path))"

# 3. /storage 挂载
echo "=== [3/7] /storage 挂载检查 ==="
ls /storage/openpsi/users/yl/agent-memory/MemRL/configs/ 2>&1 | head -5
cat /storage/openpsi/users/yl/agent-memory/MemRL/configs/rl_llb_db_memrl_haiku.yaml 2>&1 | head -3

# 4. 网络（内网 PyPI + API）
echo "=== [4/7] 网络检查 ==="
curl -s -o /dev/null -w "%{http_code}" https://pypi.antfin-inc.com/simple/ && echo " pypi.antfin-inc.com OK" || echo "pypi.antfin-inc.com FAIL"
curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com/ && echo " anthropic API OK" || echo "anthropic API FAIL (可能需要 matrixllm)"
python3 -c "
import os, urllib.request
# 检查 Anthropic API key
key = os.environ.get('ANTHROPIC_API_KEY', '')
print(f'ANTHROPIC_API_KEY set: {bool(key)} (len={len(key)})')
" 2>&1

# 5. MariaDB / MySQL 安装
echo "=== [5/7] MariaDB 安装测试 ==="
which mysqld 2>/dev/null && echo "mysqld already available" || {
    echo "mysqld not found, trying apt-get install..."
    apt-get update -qq 2>&1 | tail -3
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server 2>&1 | tail -5
    which mysqld 2>/dev/null || which mariadbd 2>/dev/null || echo "FAILED: no mysqld after install"
}

# 6. Python 依赖安装
echo "=== [6/7] Python 依赖安装 ==="
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH=${PROJECT_DIR}:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH}

pip install -e ${PROJECT_DIR} --no-deps --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm \
    concurrent-log-handler mysql-connector-python \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5

# 7. Import 验证
echo "=== [7/7] Import 验证 ==="
python3 -c "
import sys
print(f'Python: {sys.executable}')
try:
    import memos
    print(f'memos: OK ({memos.__file__})')
except ImportError as e:
    print(f'memos: FAIL ({e})')
try:
    import memrl
    print(f'memrl: OK ({memrl.__file__})')
except ImportError as e:
    print(f'memrl: FAIL ({e})')
try:
    import mysql.connector
    print(f'mysql.connector: OK')
except ImportError as e:
    print(f'mysql.connector: FAIL ({e})')
try:
    from memrl.apptainer.patch import patch_containers
    patch_containers()
    print('patch_containers(): OK')
except Exception as e:
    print(f'patch_containers(): FAIL ({e})')
"

echo "=========================================="
echo "DEBUG DONE at $(date)"
echo "Log saved to: $LOGFILE"
echo "=========================================="
