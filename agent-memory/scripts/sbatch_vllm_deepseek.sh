#!/bin/bash
#SBATCH --job-name=yl-vllm-deepseek
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/vllm_deepseek_%j.log
#SBATCH --error=logs/vllm_deepseek_%j.log

MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
VLLM_PORT=8000
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "vLLM Server: DeepSeek-R1-Distill-Qwen-32B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

echo '[INFO] Installing vLLM...'
pip install vllm --quiet 2>/dev/null || true

echo '[INFO] Starting vLLM server (TP=8)...'
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 8 \
    --port ${VLLM_PORT} \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
