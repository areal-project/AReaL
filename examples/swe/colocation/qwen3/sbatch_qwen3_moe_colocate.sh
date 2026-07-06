#!/bin/bash
#SBATCH -J zjw-qwen3-moe-colocate
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH -o ./logs/qwen3-moe-colocate-%j.out
#SBATCH -e ./logs/qwen3-moe-colocate-%j.err

set -euo pipefail

mkdir -p ./logs

# ── Paths ──
AREAL_DIR=/path/to/AReaL
IMAGE=/path/to/areal.sif

# ── Experiment ──
MODEL_PATH=/path/to/models/Qwen3-30B-A3B
EXP_NAME=zjw-qwen3-moe-colocate
TRIAL_NAME=trial-$(date +%m%d-%H%M)
N_NODES=1
N_GPUS=8

CONFIG_PATH=examples/swe/colocation/qwen3/gsm8k_grpo_qwen3_30b_colocate.yaml
FILEROOT=/path/to/experiments/qwen3-moe-colocate
NFS_ROOT=${FILEROOT}/name_resolve/${EXP_NAME}

WANDB_DIR=./wandb/${EXP_NAME}/${TRIAL_NAME}
mkdir -p "${WANDB_DIR}"

# ── Slurm ──
export AREAL_APPTAINER_STAGGER_SECONDS=5

echo "=== Job ${SLURM_JOB_ID:-local} started at $(date) on $(hostname) ==="
echo "=== Qwen3-30B-A3B MoE COLOCATE: nodes=${N_NODES} gpus=${N_GPUS} ==="
echo "=== model=${MODEL_PATH} ==="
echo "=== config=${CONFIG_PATH} ==="

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
    --env "AREAL_APPTAINER_STAGGER_SECONDS=5" \
    --env "AWEX_CHUNK_OPS=128" \
    --env "AWEX_CHUNK_MB=2048" \
    --env "AWEX_META_SERVER_ADDR=localhost:29700" \
    --env "HF_ENDPOINT=https://hf-mirror.com" \
    --env "WANDB_MODE=offline" \
    --env "WANDB_DIR=${WANDB_DIR}" \
    --env "WANDB_CACHE_DIR=${WANDB_DIR}/cache" \
    --env "WANDB_DATA_DIR=${WANDB_DIR}/data" \
    --env "WANDB_CONFIG_DIR=${WANDB_DIR}/config" \
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
        export PYTHONPATH=${AREAL_DIR}:\${PYTHONPATH:-} && \
        python3 examples/math/gsm8k_rl.py --config ${CONFIG_PATH} \
            experiment_name=${EXP_NAME} \
            actor.path=${MODEL_PATH} \
            trial_name=${TRIAL_NAME}"

EXIT_CODE=$?
echo "=== Job ${SLURM_JOB_ID:-local} finished at $(date), exit_code=${EXIT_CODE} ==="
exit ${EXIT_CODE}
