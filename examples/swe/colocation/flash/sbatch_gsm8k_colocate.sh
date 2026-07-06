#!/bin/bash
# NOTE: -J must match EXP_NAME, otherwise the job guard's prefix match fails
# and the driver cannot cancel its own worker jobs.
#SBATCH -J zjw-flash-moe-colocate
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH -o ./logs/flash-colocate-%j.out
#SBATCH -e ./logs/flash-colocate-%j.err

set -euo pipefail

mkdir -p ./logs

# ── Paths ──
AREAL_DIR=/path/to/AReaL
IMAGE=/path/to/areal.sif

# ── Experiment ──
MODEL_PATH=/path/to/models/your-moe-base-model
EXP_NAME=zjw-flash-moe-colocate
TRIAL_NAME=trial-$(date +%m%d-%H%M)
N_NODES=${N_NODES:-8}
N_GPUS=8

# Lower AWEX_CHUNK_OPS = fewer ops per batch_isend_irecv (more chunks).
AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS:-8}

CONFIG_PATH=examples/swe/colocation/flash/gsm8k_colocate.yaml
FILEROOT=/path/to/experiments/flash-moe-colocate
NFS_ROOT=${FILEROOT}/name_resolve/${EXP_NAME}

WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://your-wandb-host:8080}

export AREAL_APPTAINER_STAGGER_SECONDS=15

echo "=== Job ${SLURM_JOB_ID:-local} started at $(date) on $(hostname) ==="
echo "=== BailingMoeV2.5 Flash COLOCATE: nodes=${N_NODES} gpus=${N_GPUS} ==="
echo "=== model=${MODEL_PATH} ==="
echo "=== config=${CONFIG_PATH} ==="
echo "=== trial=${TRIAL_NAME} ==="
echo "=== AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS} (chunk-ops probe) ==="

rm -rf "${NFS_ROOT}" 2>/dev/null || true

# ── NCCL ──
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=WARN

srun --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1000M \
  singularity exec --pid --writable-tmpfs \
    --env "CONFIG_PATH=${CONFIG_PATH}" \
    --env "TRIAL_NAME=${TRIAL_NAME}" \
    --env "MODEL_PATH=${MODEL_PATH}" \
    --env "EXP_NAME=${EXP_NAME}" \
    --env "N_NODES=${N_NODES}" \
    --env "N_GPUS=${N_GPUS}" \
    --env "FILEROOT=${FILEROOT}" \
    --env "NFS_ROOT=${NFS_ROOT}" \
    --env "AREAL_APPTAINER_STAGGER_SECONDS=15" \
    --env "AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS}" \
    --env "AWEX_CHUNK_MB=2048" \
    --env "HF_ENDPOINT=https://hf-mirror.com" \
    --env "WANDB_API_KEY=${WANDB_API_KEY}" \
    --env "WANDB_BASE_URL=${WANDB_BASE_URL}" \
    --bind /storage:/storage \
    --bind /home:/home \
    --bind /etc/slurm/:/etc/slurm/ \
    --bind /etc/passwd:/etc/passwd:ro \
    --bind /etc/group:/etc/group:ro \
    --bind /etc/munge:/etc/munge:ro \
    --bind /var/run/munge:/var/run/munge \
    --bind /usr/bin/sbatch:/usr/bin/sbatch \
    --bind /usr/bin/srun:/usr/bin/srun \
    --bind /usr/bin/squeue:/usr/bin/squeue \
    --bind /usr/bin/scancel:/usr/bin/scancel \
    --bind /usr/bin/scontrol:/usr/bin/scontrol \
    --bind /usr/lib64/slurm:/usr/lib64/slurm \
    "${IMAGE}" \
    bash -c "(/usr/sbin/munged 2>/dev/null || true) && \
        cd ${AREAL_DIR} && \
                uv pip install -e . && \
        export PYTHONPATH=${AREAL_DIR}:/path/to/awex:/path/to/flash-linear-attention:\${PYTHONPATH:-} && \
        python3 examples/math/gsm8k_rl.py --config ${CONFIG_PATH} \
            experiment_name=${EXP_NAME} \
            actor.path=${MODEL_PATH} \
            saver.freq_steps=${SAVER_FREQ_STEPS:-40} \
            trial_name=${TRIAL_NAME}"

EXIT_CODE=$?
echo "=== Job ${SLURM_JOB_ID:-local} finished at $(date), exit_code=${EXIT_CODE} ==="
exit ${EXIT_CODE}
