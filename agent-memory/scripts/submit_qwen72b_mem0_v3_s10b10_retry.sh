#!/bin/bash
# Submit isolated Qwen2.5-72B ALFWorld Mem0 continuation retry from the isolated S10 batch 10 checkpoint.
set -euo pipefail

export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="477578"
export AISTUDIO_TOKEN="7371e433-4755-44b1-b410-319ab4024990"
export WORKER_NUM=0
export JOB_TAG="qwen72b-mem0-v3-s10b10-retry"
TS="20260729_105517"
export JOB_NAME="yl-alf-q72b-mem0-v3-s10b10-retry-$(date +%Y%m%d_%H%M%S)"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"
# MEMRL_RUN_ID must remain stable: Mem0 retries restore from the destination
# snapshot tree rather than generic ckpt_resume_path.
export JOB_COMMAND="MEMRL_RUN_ID=qwen72b_mem0_v3_s9b10_resume_20260729_105517 MEM0_COLLECTION=memrl_mem0_alf_qwen72b_v3 MEM0_RESUME_SOURCE=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_mem0_qwen72b_qwen72b_mem0_v3/local_cache/snapshot/s9_b10 MEM0_EXPECTED_RESUME=section=10,batch=11 MEM0_CKPT_RESUME_ENABLED=true EMBED_PORT=19090 LLM_PORT=19290 NCCL_PORT=29690 bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_alfworld_qwen72b_mem0_vllm.sh ${TS}"
export GPU_NUM=3
export GPU_TYPE="h200"

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
export LAUNCH_CONTAINER_MODE=dev_local

aistudio_user login --name "$AISTUDIO_LOGIN_NAME" --usernumber "$AISTUDIO_USERNUMBER" --token "$AISTUDIO_TOKEN"
cd /tmp
PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_regionv2.py
