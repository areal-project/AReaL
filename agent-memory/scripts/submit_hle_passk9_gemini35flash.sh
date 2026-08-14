#!/bin/bash
# ============================================================================
# HLE pass@9 baseline: gemini-3.5-flash via MatrixLLM API
# Combined with nomem (round 1), this gives pass@10
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
export JOB_NAME="yl-hle-passk9-g35f-${TS}"

export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# ---------------------------------------------------------------------------
# 启动命令
# ---------------------------------------------------------------------------
export JOB_COMMAND="bash -c '
set -e
LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_hle_passk9_gemini35flash_${TS}.log
exec > >(tee -a \$LOGFILE) 2>&1

echo \"=== Start: \$(date) ===\"
cd /storage/openpsi/users/yl/agent-memory/MemRL

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai \"chonkie==1.2.1\" tensorboard pandas tqdm concurrent-log-handler ollama --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:\$PYTHONPATH

python3 -c \"import memos; import memrl; print(\\\"All imports OK\\\")\"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_MIN_INTERVAL=1.0
export MEMRL_REASONING_EFFORT=high

# Fix run_id so we can pre-create the log dir and seed round 1
export MEMRL_RUN_ID=passk10_from_nomem
CKPT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_passk9_gemini35flash_\${MEMRL_RUN_ID}/local_cache
mkdir -p \$CKPT_DIR

# Seed round 1 from completed nomem experiment
NOMEM_CKPT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_nomem_gemini35flash_20260706-144723/local_cache
python3 scripts/seed_passk_round1.py --nomem-ckpt \$NOMEM_CKPT --passk-log-dir \$CKPT_DIR

python3 run/run_hle.py \
    --config configs/rl_hle_config.passk9_gemini35flash.yaml \
    --train data/hle/hle_test.parquet \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5

echo \"=== Done: \$(date) ===\"
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
