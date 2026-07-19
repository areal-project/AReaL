#!/usr/bin/env bash
#SBATCH -J swe-arena-stream
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -t 7-00:00:00
#SBATCH -o rl_logs/swe-arena-stream-%j.out
#SBATCH -e rl_logs/swe-arena-stream-%j.err

set -euo pipefail

: "${AREAL_DIR:?AREAL_DIR must point to this AReaL checkout}"
: "${AREAL_IMAGE:?AREAL_IMAGE must point to the training image}"
: "${AREAL_FILEROOT:?AREAL_FILEROOT must be shared storage}"
: "${AREAL_CACHE_ROOT:?AREAL_CACHE_ROOT must be shared storage}"
: "${MODEL_PATH:?MODEL_PATH must point to the Qwen3-Coder checkpoint}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${TRIAL_NAME:?TRIAL_NAME is required}"
: "${ARENA_OPENAPI_BASE:?ARENA_OPENAPI_BASE is required}"
: "${ARENA_OPENAPI_TOKEN:?ARENA_OPENAPI_TOKEN is required}"
: "${ARENA_LLM_API_KEY:?ARENA_LLM_API_KEY is required}"
: "${SWE_RL_ADMIN_API_KEY:?SWE_RL_ADMIN_API_KEY is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${WANDB_BASE_URL:?WANDB_BASE_URL is required}"

AREAL_PYTHON="${AREAL_PYTHON:-/opt/.venv/bin/python3}"

mkdir -p \
  "${AREAL_DIR}/rl_logs" \
  "${AREAL_FILEROOT}/name_resolve/${EXPERIMENT_NAME}" \
  "${AREAL_CACHE_ROOT}/home" \
  "${AREAL_CACHE_ROOT}/hf" \
  "${AREAL_CACHE_ROOT}/xdg"

export APPTAINERENV_AREAL_DIR="${AREAL_DIR}"
export APPTAINERENV_AREAL_IMAGE="${AREAL_IMAGE}"
export APPTAINERENV_AREAL_FILEROOT="${AREAL_FILEROOT}"
export APPTAINERENV_AREAL_CACHE_ROOT="${AREAL_CACHE_ROOT}"
export APPTAINERENV_MODEL_PATH="${MODEL_PATH}"
export APPTAINERENV_EXPERIMENT_NAME="${EXPERIMENT_NAME}"
export APPTAINERENV_TRIAL_NAME="${TRIAL_NAME}"
export APPTAINERENV_ARENA_OPENAPI_BASE="${ARENA_OPENAPI_BASE}"
export APPTAINERENV_ARENA_OPENAPI_TOKEN="${ARENA_OPENAPI_TOKEN}"
export APPTAINERENV_ARENA_LLM_API_KEY="${ARENA_LLM_API_KEY}"
export APPTAINERENV_SWE_RL_ADMIN_API_KEY="${SWE_RL_ADMIN_API_KEY}"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_BASE_URL="${WANDB_BASE_URL}"
export APPTAINERENV_AREAL_PYTHON="${AREAL_PYTHON}"
export APPTAINERENV_HOME="${AREAL_CACHE_ROOT}/home"
export APPTAINERENV_HF_HOME="${AREAL_CACHE_ROOT}/hf"
export APPTAINERENV_XDG_CACHE_HOME="${AREAL_CACHE_ROOT}/xdg"
export APPTAINERENV_PYTHONPATH="${AREAL_DIR}"

echo "Starting Arena Stream controller job ${SLURM_JOB_ID}"

srun --mpi=pmi2 --ntasks=1 --cpus-per-task=4 --mem=10G \
  singularity exec --pid --writable-tmpfs \
    --bind /storage:/storage \
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
    "${AREAL_IMAGE}" \
    bash -lc "
      (/usr/sbin/munged 2>/dev/null || true)
      cd '${AREAL_DIR}'
      '${AREAL_PYTHON}' -m examples.swe.train_swe_rl \
        --config examples/swe/qwen3_30b_a3b_grpo.yaml
    "
