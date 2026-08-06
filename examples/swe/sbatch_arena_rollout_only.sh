#!/usr/bin/env bash
#SBATCH -J swe-arena-rollout-only
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -o rl_logs/swe-arena-rollout-only-%j.out
#SBATCH -e rl_logs/swe-arena-rollout-only-%j.err

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
ARENA_NUM_ROLLOUTS="${ARENA_NUM_ROLLOUTS:-16}"
ARENA_MAX_CONCURRENT_ROLLOUTS="${ARENA_MAX_CONCURRENT_ROLLOUTS:-4}"
ROLLOUT_ONLY_MODE_ARG=""
if [[ "${ARENA_REGISTRY_SMOKE:-0}" == "1" ]]; then
  ROLLOUT_ONLY_MODE_ARG="--registry-smoke"
fi

mkdir -p \
  "${AREAL_DIR}/rl_logs" \
  "${AREAL_FILEROOT}/name_resolve/${EXPERIMENT_NAME}" \
  "${AREAL_CACHE_ROOT}/home" \
  "${AREAL_CACHE_ROOT}/hf" \
  "${AREAL_CACHE_ROOT}/xdg"

export APPTAINERENV_AREAL_DIR="${AREAL_DIR}"
export APPTAINERENV_AREAL_FILEROOT="${AREAL_FILEROOT}"
export APPTAINERENV_AREAL_CACHE_ROOT="${AREAL_CACHE_ROOT}"
export APPTAINERENV_MODEL_PATH="${MODEL_PATH}"
export APPTAINERENV_EXPERIMENT_NAME="${EXPERIMENT_NAME}"
export APPTAINERENV_TRIAL_NAME="${TRIAL_NAME}"
export APPTAINERENV_ARENA_OPENAPI_BASE="${ARENA_OPENAPI_BASE}"
export APPTAINERENV_ARENA_OPENAPI_TOKEN="${ARENA_OPENAPI_TOKEN}"
export APPTAINERENV_ARENA_LLM_API_KEY="${ARENA_LLM_API_KEY:-}"
export APPTAINERENV_ARENA_ROLLOUT_DATA_ID="${ARENA_ROLLOUT_DATA_ID:-}"
export APPTAINERENV_SWE_RL_ADMIN_API_KEY="${SWE_RL_ADMIN_API_KEY}"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_BASE_URL="${WANDB_BASE_URL}"
export APPTAINERENV_AREAL_PYTHON="${AREAL_PYTHON}"
export APPTAINERENV_HOME="${AREAL_CACHE_ROOT}/home"
export APPTAINERENV_HF_HOME="${AREAL_CACHE_ROOT}/hf"
export APPTAINERENV_XDG_CACHE_HOME="${AREAL_CACHE_ROOT}/xdg"
export APPTAINERENV_PYTHONPATH="${AREAL_DIR}"

echo "Starting single-node Arena rollout-only job ${SLURM_JOB_ID}"

srun --mpi=pmi2 --ntasks=1 --cpus-per-task=32 --mem=256G \
  singularity exec --nv --pid --writable-tmpfs \
    --bind /storage:/storage \
    "${AREAL_IMAGE}" \
    bash -lc "
      export PATH='${AREAL_PYTHON%/*}':\${PATH}
      cd '${AREAL_DIR}'
      '${AREAL_PYTHON}' -m examples.swe.arena_rollout_only \
        ${ROLLOUT_ONLY_MODE_ARG} \
        --num-rollouts "${ARENA_NUM_ROLLOUTS}" \
        --config examples/swe/qwen3_30b_a3b_grpo.yaml \
        scheduler.type=local \
        cluster.n_nodes=1 \
        cluster.n_gpus_per_node=8 \
        rollout.backend=sglang:d1t8p1 \
        gconfig.n_samples=1 \
        rollout.consumer_batch_size="${ARENA_NUM_ROLLOUTS}" \
        rollout.max_concurrent_rollouts="${ARENA_MAX_CONCURRENT_ROLLOUTS}" \
        rollout.setup_timeout=3600.0 \
        sglang.context_length="${SGLANG_CONTEXT_LENGTH:-133120}" \
        sglang.max_prefill_tokens="${SGLANG_MAX_PREFILL_TOKENS:-133119}" \
        sglang.max_running_requests="${ARENA_MAX_CONCURRENT_ROLLOUTS}" \
        sglang.cuda_graph_max_bs="${ARENA_MAX_CONCURRENT_ROLLOUTS}" \
        econfig.arena_request_timeout=30.0 \
        train_dataset.batch_size="${ARENA_NUM_ROLLOUTS}" \
        train_dataset.num_workers=0 \
        valid_dataset.batch_size="${ARENA_NUM_ROLLOUTS}" \
        valid_dataset.num_workers=0
    "
