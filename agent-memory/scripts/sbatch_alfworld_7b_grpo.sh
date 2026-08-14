#!/bin/bash
#SBATCH --job-name=yl-alf-7b-grpo
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_7b_grpo_%j.log
#SBATCH --error=logs/alf_7b_grpo_%j.log

# ALFWorld 7B GRPO Training — dual-container env-server architecture.
# GPU 0-4 occupied by yl-alf-fs-abl, use GPU 5,6.

AREAL_DIR="/storage/openpsi/users/yl/AReaL"
ENV_IMG="/storage/openpsi/images/areal-latest.sif"
TRAINER_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
GPUS="2,3"
ENV_PORT=8765

echo "=========================================="
echo "ALFWorld 7B GRPO Training (env-server arch)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

cleanup() {
    echo "[CLEANUP] Killing env server (PID=$ENV_PID)..."
    kill $ENV_PID 2>/dev/null
    wait $ENV_PID 2>/dev/null
}
trap cleanup EXIT

# === Step 1: Start env server in areal-latest.sif ===
echo "[1/3] Starting env server on port ${ENV_PORT}..."
singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $ENV_IMG \
    bash -c "
pip install fastapi uvicorn textworld alfworld --quiet 2>&1 | tail -3
cd ${AREAL_DIR}
export PYTHONPATH=${AREAL_DIR}:\$PYTHONPATH
echo \"[env_server] Starting (process-per-env, no restart needed)...\"
python examples/alfworld/env_server.py --port ${ENV_PORT}
" &
ENV_PID=$!
echo "Env server PID: $ENV_PID"

# Wait for env server to be ready
echo "[1/3] Waiting for env server..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${ENV_PORT}/health" > /dev/null 2>&1; then
        echo "[1/3] Env server ready!"
        break
    fi
    if ! kill -0 $ENV_PID 2>/dev/null; then
        echo "[ERROR] Env server process died!"
        exit 1
    fi
    sleep 2
done

if ! curl -s "http://127.0.0.1:${ENV_PORT}/health" > /dev/null 2>&1; then
    echo "[ERROR] Env server failed to start after 120s"
    exit 1
fi

# === Step 2: Build dataset (in trainer image) ===
echo "[2/3] Building dataset + starting trainer..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $TRAINER_IMG \
    bash -c "
cd ${AREAL_DIR}
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENV_SERVER_URL=http://127.0.0.1:${ENV_PORT}
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${AREAL_DIR}:\$PYTHONPATH

echo '[INFO] Using GPUs: ${GPUS}'
echo '[INFO] ENV_SERVER_URL: \$ENV_SERVER_URL'

echo '[INFO] Verifying imports...'
python -c 'from areal import PPOTrainer; print(\"PPOTrainer OK\")' || { echo '[ERROR] PPOTrainer import failed'; exit 1; }
python -c 'from examples.alfworld.env_client import get_client; print(\"env_client OK\")' || { echo '[ERROR] env_client import failed'; exit 1; }

echo '[INFO] Building dataset...'
pip install datasets 2>&1 | tail -3
python examples/alfworld/dataset.py \
    --data_root /storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1 \
    --output_dir /tmp/areal/alfworld_dataset \
    --split train

echo ''
echo '[RUN] GRPO Training (1 epoch)'
python examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    scheduler.type=local
echo \"[7b grpo] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
