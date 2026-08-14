#!/bin/bash
# ============================================================================
# HLE RAG + Self-RAG 合并: 一张 GPU 容器内并行跑两个实验
# 纯 API 调用，不用 GPU，节省资源；两者存储/检索机制近似，共用一台机器
# ============================================================================

# ---------------------------------------------------------------------------
# 登录凭证（aistudio）
# ---------------------------------------------------------------------------
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"

# ---------------------------------------------------------------------------
# 作业信息
# ---------------------------------------------------------------------------
export WORKER_NUM="0"
export JOB_TAG=""
TS=$(date +%Y%m%d-%H%M%S)
export JOB_NAME="${JOB_NAME:-yl-hle-rag-selfrag-${TS}}"

export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# ---------------------------------------------------------------------------
# Run IDs (new runs; override MEMRL_RUN_ID_RAG / MEMRL_RUN_ID_SELFRAG to resume)
# ---------------------------------------------------------------------------
RAG_RID="${MEMRL_RUN_ID_RAG:-yl-hle-rag-g35f-${TS}}"
SELFRAG_RID="${MEMRL_RUN_ID_SELFRAG:-yl-hle-selfrag-g35f-${TS}}"

# ---------------------------------------------------------------------------
# 启动命令
# ---------------------------------------------------------------------------
export JOB_COMMAND="bash -c '
set -e

echo \"=== Start: \$(date) ===\"
cd /storage/openpsi/users/yl/agent-memory/MemRL

# Dependencies
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai \"chonkie==1.2.1\" tensorboard pandas tqdm concurrent-log-handler ollama --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:\$PYTHONPATH
python3 -c \"import memos; import memrl; print(\\\"All imports OK\\\")\"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Common settings
export MEMRL_REASONING_EFFORT=high
export MEMRL_RUNNER_MAX_RETRIES=1

# --- RAG process ---
(
    export MEMRL_RUN_ID=\"${RAG_RID}\"
    export MEMRL_LLM_MIN_INTERVAL=1.0
    export MEMRL_EMBED_MIN_INTERVAL=1.0
    export MEMRL_UPDATE_MAX_WORKERS=4
    LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_hle_rag_gemini35flash_${RAG_RID}.log
    exec > >(tee -a \$LOGFILE) 2>&1
    echo \"[RAG] Start: \$(date), RUN_ID=${RAG_RID}\"
    python3 run/run_hle.py \
        --config configs/rl_hle_config.rag_gemini35flash.yaml \
        --train data/hle/hle_test.parquet \
        --judge_model gpt-4o-2024-11-20 \
        --judge_base_url https://matrixllm.alipay.com/v1/ \
        --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
    echo \"[RAG] Done: \$(date)\"
) &
RAG_PID=\$!

echo \"Sleeping 15 min before starting Self-RAG to stagger memory update...\"
sleep 900

# --- Self-RAG process ---
(
    export MEMRL_RUN_ID=\"${SELFRAG_RID}\"
    export MEMRL_LLM_MIN_INTERVAL=2.0
    export MEMRL_EMBED_MIN_INTERVAL=2.0
    export MEMRL_UPDATE_MAX_WORKERS=4
    LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_hle_selfrag_gemini35flash_${SELFRAG_RID}.log
    exec > >(tee -a \$LOGFILE) 2>&1
    echo \"[Self-RAG] Start: \$(date), RUN_ID=${SELFRAG_RID}\"
    python3 run/run_hle.py \
        --config configs/rl_hle_config.selfrag_gemini35flash.yaml \
        --train data/hle/hle_test.parquet \
        --self_rag \
        --judge_model gpt-4o-2024-11-20 \
        --judge_base_url https://matrixllm.alipay.com/v1/ \
        --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
    echo \"[Self-RAG] Done: \$(date)\"
) &
SELFRAG_PID=\$!

echo \"Launched RAG (PID=\$RAG_PID) and Self-RAG (PID=\$SELFRAG_PID)\"
wait \$RAG_PID \$SELFRAG_PID
echo \"=== All Done: \$(date) ===\"
'"

# ---------------------------------------------------------------------------
# 环境准备与提交
# ---------------------------------------------------------------------------
if [ -z "${AISTUDIO_USERNUMBER}" ]; then
    echo "错误：AISTUDIO_USERNUMBER 为空" >&2
    exit 1
fi

if [ -z "${AISTUDIO_TOKEN}" ]; then
    echo "错误：AISTUDIO_TOKEN 为空" >&2
    exit 1
fi

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/

export LAUNCH_CONTAINER_MODE=dev_local

aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"

python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_hle_template.py
