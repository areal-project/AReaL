#!/bin/bash
# ============================================================================
# HLE MemRL: gemini-3.5-flash via MatrixLLM API
# 纯 API 调用，不需要 GPU
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
export JOB_NAME="yl-hle-memrl-g35f-resume064537-${TS}"

export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"

# ---------------------------------------------------------------------------
# 启动命令
# ---------------------------------------------------------------------------
export JOB_COMMAND="bash -c '
set -e
LOGFILE=/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_hle_memrl_gemini35flash_${TS}.log
exec > >(tee -a \$LOGFILE) 2>&1

echo \"=== Start: \$(date) ===\"
cd /storage/openpsi/users/yl/agent-memory/MemRL

# Pin ckpt dir name to submit-time TS so a platform retry (RetryPolicy.ON_FAILURE)
# writes to the SAME exp_<name>_<run_id>/ dir -> _resume_from_ckpt_if_needed picks up
# the existing snapshot and resumes from the last batch ckpt instead of restarting.
export MEMRL_RUN_ID="20260707-064537"  # resume from sec1/batch21 (704/2500) of the preempted first run

# Use source directly via PYTHONPATH (no pip install needed for memrl).
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai \"chonkie==1.2.1\" tensorboard pandas tqdm concurrent-log-handler ollama --target \$VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3

export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:\$PYTHONPATH

# Verify imports
python3 -c \"import memos; import memrl; print(\\\"All imports OK\\\")\"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_MIN_INTERVAL=1.0
export MEMRL_UPDATE_MAX_WORKERS=4
export MEMRL_REASONING_EFFORT=high

python3 run/run_hle.py \
    --config configs/rl_hle_config.memrl_gemini35flash.yaml \
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
