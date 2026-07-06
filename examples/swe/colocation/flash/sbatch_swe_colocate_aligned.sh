#!/bin/bash
# SWE RL colocated long-run on the flash cluster: 8 nodes, actor+rollout
# colocated via AWEX weight sync. Runs examples/swe/train_swe_rl.py with the
# AReaL-SWEAgent workflow (aweagent).
#SBATCH -J zjw-swe-colocate-aligned
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH -t 7-00:00:00
#SBATCH -o ./logs/swe-colocate-aligned-%j.out
#SBATCH -e ./logs/swe-colocate-aligned-%j.err

set -euo pipefail
mkdir -p ./logs

AREAL_DIR=/path/to/AReaL
AWEX_DIR=/path/to/awex
SGLANG_DIR=/path/to/sglang/python
FLA_DIR=/path/to/flash-linear-attention
IMAGE=/path/to/areal.sif

MODEL_PATH=${MODEL_PATH:-/path/to/models/your-swe-sft-model}
TRAIN_DATA=${TRAIN_DATA:-/path/to/data/swe_bench_rl.jsonl}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-/path/to/AReaL-SWEAgent}
EXP_NAME=${EXP_NAME:-zjw-swe-colocate-aligned}
TRIAL_NAME=${TRIAL_NAME:-stability-8n-$(date +%m%d-%H%M)}
N_NODES=${N_NODES:-8}
N_GPUS=8

CONFIG_PATH=examples/swe/colocation/flash/swe_colocate_aligned.yaml
FILEROOT=/path/to/experiments/flash-moe-colocate
NFS_ROOT=${FILEROOT}/name_resolve/${EXP_NAME}

AENV_SYSTEM_URL=${AENV_SYSTEM_URL:-http://33.180.184.68}
WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://your-wandb-host:8080}

export AREAL_APPTAINER_STAGGER_SECONDS=15

echo "=== zjw SWE COLOCATE ALIGNED nodes=${N_NODES} trial=${TRIAL_NAME} ==="
echo "=== areal=${AREAL_DIR} model=${MODEL_PATH} aweagent=${AWEAGENT_ROOT} config=${CONFIG_PATH} ==="

rm -rf "${NFS_ROOT}" 2>/dev/null || true

export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=WARN

MAX_DRIVER_RETRIES=${MAX_DRIVER_RETRIES:-5}
attempt=1
while true; do
echo "=== driver attempt ${attempt}/${MAX_DRIVER_RETRIES} trial ${TRIAL_NAME} $(date) ==="
set +e
srun --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1000M \
  singularity exec --pid --writable-tmpfs \
    --env "CONFIG_PATH=${CONFIG_PATH}" \
    --env "TRIAL_NAME=${TRIAL_NAME}" \
    --env "MODEL_PATH=${MODEL_PATH}" \
    --env "TRAIN_DATA=${TRAIN_DATA}" \
    --env "EXP_NAME=${EXP_NAME}" \
    --env "N_NODES=${N_NODES}" \
    --env "N_GPUS=${N_GPUS}" \
    --env "FILEROOT=${FILEROOT}" \
    --env "NFS_ROOT=${NFS_ROOT}" \
    --env "AWEAGENT_ROOT=${AWEAGENT_ROOT}" \
    --env "SWE_AGENT_ROOT=${AWEAGENT_ROOT}" \
    --env "AENV_SYSTEM_URL=${AENV_SYSTEM_URL}" \
    --env "AREAL_APPTAINER_STAGGER_SECONDS=15" \
    --env "AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS:-128}" \
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
        export PYTHONPATH=${AREAL_DIR}:${AWEX_DIR}:${SGLANG_DIR}:${FLA_DIR}:\${PYTHONPATH:-} && \
        python3 examples/swe/train_swe_rl.py --config ${CONFIG_PATH} \
            experiment_name=${EXP_NAME} \
            trial_name=${TRIAL_NAME} \
            tokenizer_path=${MODEL_PATH} \
            actor.path=${MODEL_PATH} \
            cluster.n_nodes=${N_NODES} \
            cluster.n_gpus_per_node=${N_GPUS} \
            cluster.fileroot=${FILEROOT} \
            cluster.name_resolve.nfs_record_root=${NFS_ROOT} \
            train_dataset.path=${TRAIN_DATA} \
            valid_dataset.path=${TRAIN_DATA} \
            stats_logger.wandb.mode=online \
            ++stats_logger.wandb.wandb_api_key=${WANDB_API_KEY} \
            ++stats_logger.wandb.wandb_base_url=${WANDB_BASE_URL} \
            \"++actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL=${AENV_SYSTEM_URL}\" \
            \"++actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT=${AWEAGENT_ROOT}\" \
            \"++actor.scheduling_spec.0.env_vars.SWE_AGENT_ROOT=${AWEAGENT_ROOT}\" \
            \"++actor.scheduling_spec.0.env_vars.WANDB_BASE_URL=${WANDB_BASE_URL}\" \
            \"++actor.scheduling_spec.0.env_vars.WANDB_API_KEY=${WANDB_API_KEY}\""

EXIT_CODE=$?
set -e
if [ ${EXIT_CODE} -eq 0 ]; then break; fi
if [ ${attempt} -ge ${MAX_DRIVER_RETRIES} ]; then
  echo "=== driver failed ${MAX_DRIVER_RETRIES} times, giving up ==="
  break
fi
attempt=$((attempt+1))
echo "=== driver exited ${EXIT_CODE}; wait 90s then recover trial ${TRIAL_NAME} ==="
sleep 90
done

echo "=== Job ${SLURM_JOB_ID:-local} finished $(date), exit=${EXIT_CODE} ==="
exit ${EXIT_CODE}
