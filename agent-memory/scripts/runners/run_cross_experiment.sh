#!/bin/bash
# Cross-benchmark 实验运行脚本
# 在计算节点上启动 LiteLLM 服务并运行实验
# 用法: run_cross_experiment.sh <source> <targets> <experiment_name>

set -e

# Disable Python bytecode to avoid null byte issues
export PYTHONDONTWRITEBYTECODE=1

SOURCE=$1
TARGETS=$2
EXP_NAME=$3

cd /storage/openpsi/users/yl/agent-memory/MemRL

# 安装依赖
echo "[INFO] 安装项目依赖..."
pip install -e . -q 2>/dev/null
pip install -r requirements.txt -q 2>/dev/null

# 安装 LiteLLM
echo "[INFO] 安装 LiteLLM..."
pip install 'litellm[proxy]' -q 2>/dev/null

# 选择一个随机可用端口 (避免冲突)
LITELLM_PORT=$((10000 + RANDOM % 50000))
echo "[INFO] 使用端口: ${LITELLM_PORT}"

# 创建带端口配置的临时配置文件
TEMP_CONFIG="/tmp/litellm_config_${EXP_NAME}.yaml"
cp /storage/openpsi/users/yl/.claude/config.yaml ${TEMP_CONFIG}

# 在后台启动 LiteLLM 服务
echo "[INFO] 启动 LiteLLM 服务..."
python -m litellm.proxy.proxy_cli --config ${TEMP_CONFIG} --port ${LITELLM_PORT} --host 127.0.0.1 > /tmp/litellm_${EXP_NAME}.log 2>&1 &

LITELLM_PID=$!
echo "[INFO] LiteLLM PID: ${LITELLM_PID}"

# 等待 LiteLLM 启动
echo "[INFO] 等待 LiteLLM 服务就绪..."
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}"
MAX_RETRIES=60
for i in $(seq 1 ${MAX_RETRIES}); do
    if curl -s ${LITELLM_URL}/health > /dev/null 2>&1; then
        echo "[INFO] LiteLLM 服务已就绪 (尝试 $i)"
        break
    fi
    # 检查进程是否还在运行
    if ! kill -0 ${LITELLM_PID} 2>/dev/null; then
        echo "[ERROR] LiteLLM 进程已退出，查看日志:"
        tail -50 /tmp/litellm_${EXP_NAME}.log
        exit 1
    fi
    if [ $i -eq ${MAX_RETRIES} ]; then
        echo "[ERROR] LiteLLM 服务启动超时，查看日志:"
        tail -100 /tmp/litellm_${EXP_NAME}.log
        kill ${LITELLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 2
done

# 验证服务
echo "[INFO] 验证 LiteLLM 服务..."
curl -s ${LITELLM_URL}/health | head -c 100
echo ""

# 运行实验
echo "[INFO] 开始运行实验: ${EXP_NAME}"
echo "[INFO] Source: ${SOURCE}, Targets: ${TARGETS}"
echo "[INFO] LiteLLM URL: ${LITELLM_URL}"

# Use python3 -B to avoid .pyc files which can cause null byte issues
python3 -B scripts/run_cross_benchmark_experiment.py \
    --source ${SOURCE} \
    --targets ${TARGETS} \
    --api_key 'sk-placeholder' \
    --base_url "${LITELLM_URL}" \
    --model 'gpt-4o-2024-11-20' \
    --embedding_model 'text-embedding-3-small' \
    --epochs 5 \
    --batch_size 5 \
    --name "${EXP_NAME}" \
    --mode local

EXIT_CODE=$?

# 清理
echo "[INFO] 停止 LiteLLM 服务..."
kill ${LITELLM_PID} 2>/dev/null
rm -f ${TEMP_CONFIG}

echo "[INFO] 实验完成，退出码: ${EXIT_CODE}"
exit ${EXIT_CODE}
