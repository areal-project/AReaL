#!/bin/bash
#SBATCH --job-name=yl-alf-7b-eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_7b_eval_%j.log
#SBATCH --error=logs/alf_7b_eval_%j.log

# ALFWorld 7B GRPO Eval — no memory, all splits (train/val/ood).
# Serves ckpt with vLLM + env_server, runs eval.py.

AREAL_DIR="/storage/openpsi/users/yl/AReaL"
ENV_IMG="/storage/openpsi/images/areal-latest.sif"
TRAINER_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-areal-rl/checkpoints/admin/alfworld-7b-grpo-inline-fs/trial0/default/epoch0epochstep443globalstep443"
GPU="2"
ENV_PORT=8765
VLLM_PORT=8000
MAX_EPISODES=9999

echo "=========================================="
echo "ALFWorld 7B GRPO Eval (no memory, all splits)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "Model: $CKPT"
echo "=========================================="

cleanup() {
    echo "[CLEANUP] Killing env server (PID=$ENV_PID) and vLLM (PID=$VLLM_PID)..."
    kill $ENV_PID $VLLM_PID 2>/dev/null
    wait $ENV_PID $VLLM_PID 2>/dev/null
}
trap cleanup EXIT

# === Step 1: Start env server ===
echo "[1/3] Starting env server on port ${ENV_PORT}..."
singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $ENV_IMG \
    bash -c "
pip install fastapi uvicorn textworld alfworld --quiet 2>&1 | tail -3
cd ${AREAL_DIR}
export PYTHONPATH=${AREAL_DIR}:\$PYTHONPATH
python examples/alfworld/env_server.py --port ${ENV_PORT}
" &
ENV_PID=$!

# Wait for env server
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${ENV_PORT}/health" > /dev/null 2>&1; then
        echo "[1/3] Env server ready!"
        break
    fi
    sleep 2
done

# === Step 2: Start vLLM server ===
echo "[2/3] Starting vLLM server with checkpoint..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $TRAINER_IMG \
    bash -c "
export CUDA_VISIBLE_DEVICES=${GPU}
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
python -m vllm.entrypoints.openai.api_server \
    --model ${CKPT} \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --served-model-name default \
    --trust-remote-code
" &
VLLM_PID=$!

# Wait for vLLM
echo "[2/3] Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[2/3] vLLM server ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] vLLM server died!"
        exit 1
    fi
    sleep 3
done

if ! curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM server failed to start"
    exit 1
fi

# === Step 3: Run eval ===
echo "[3/3] Running eval on train + valid_seen + valid_unseen..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $TRAINER_IMG \
    bash -c "
cd ${AREAL_DIR}
export PYTHONPATH=${AREAL_DIR}:\$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
pip install aiohttp --quiet 2>&1 | tail -1

python examples/alfworld/eval.py \
    --model_path ${CKPT} \
    --splits train valid_seen valid_unseen \
    --max_episodes ${MAX_EPISODES} \
    --max_steps 30 \
    --concurrency 16 \
    --vllm_port ${VLLM_PORT} \
    --env_port ${ENV_PORT} \
    --output /storage/openpsi/users/yl/AReaL/examples/alfworld/eval_results_7b_inline_fs.json

echo \"[eval] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
