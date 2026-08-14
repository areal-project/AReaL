#!/bin/bash
#SBATCH --job-name=yl-alf-7b-region-fs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_7b_region_fs_%j.log
#SBATCH --error=logs/alf_7b_region_fs_%j.log

# ALFWorld 7B GRPO + Region Failure Summary Training.
# Same as baseline but with region FS injected into prompts.

AREAL_DIR="/storage/openpsi/users/yl/AReaL"
ENV_IMG="/storage/openpsi/images/areal-latest.sif"
TRAINER_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
GPUS="2,3"
ENV_PORT=8766

echo "=========================================="
echo "ALFWorld 7B GRPO + Region FS Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

cleanup() {
    echo "[CLEANUP] Killing env server (PID=$ENV_PID)..."
    kill $ENV_PID 2>/dev/null
    wait $ENV_PID 2>/dev/null
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

if ! curl -s "http://127.0.0.1:${ENV_PORT}/health" > /dev/null 2>&1; then
    echo "[ERROR] Env server failed to start"
    exit 1
fi

# === Step 2: Build dataset with region FS + start trainer ===
echo "[2/3] Building dataset (with region FS) + starting trainer..."
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
echo '[INFO] Building dataset with region failure summaries...'
pip install datasets 2>&1 | tail -3
python examples/alfworld/dataset.py \
    --data_root /storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1 \
    --output_dir /tmp/areal/alfworld_dataset_region_fs \
    --split train \
    --failure_summary ${AREAL_DIR}/examples/alfworld/game_file_to_onpolicy_fs.json \
    --fs_ratio 0.5

echo ''
echo '[RUN] GRPO Training with Region FS (1 epoch)'
python examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    scheduler.type=local \
    experiment_name=alfworld-7b-grpo-region-fs \
    train_dataset.path=/tmp/areal/alfworld_dataset_region_fs/combined \
    valid_dataset.path=/tmp/areal/alfworld_dataset_region_fs/combined \
    actor.optimizer.lr=5e-6 \
    actor.optimizer.warmup_steps_proportion=0.08 \
    actor.optimizer.lr_scheduler_type=constant \
    actor.eps_clip=0.15 \
    actor.kl_ctl=0.03 \
    actor.reward_scaling=7.0
echo \"[7b grpo region-fs] exit=\$? at \$(date)\"
"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
