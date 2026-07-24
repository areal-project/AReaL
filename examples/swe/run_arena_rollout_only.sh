#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AREAL_DIR="${AREAL_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ARENA_RUN_ENV_FILE="${ARENA_RUN_ENV_FILE:-${AREAL_DIR}/.arena-rollout.env}"

if [[ ! -f "${ARENA_RUN_ENV_FILE}" ]]; then
  echo "Missing ${ARENA_RUN_ENV_FILE}. Create it or set ARENA_RUN_ENV_FILE." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ARENA_RUN_ENV_FILE}"
set +a

export AREAL_DIR
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe-arena-rollout-only}"
export TRIAL_NAME="${TRIAL_NAME:-qwen3-coder-30b-a3b-swebench-128k-batch16-concurrency4-$(date +%Y%m%d-%H%M%S)}"
export ARENA_NUM_ROLLOUTS="${ARENA_NUM_ROLLOUTS:-16}"
export ARENA_MAX_CONCURRENT_ROLLOUTS="${ARENA_MAX_CONCURRENT_ROLLOUTS:-4}"

required_variables=(
  AREAL_IMAGE
  AREAL_FILEROOT
  AREAL_CACHE_ROOT
  MODEL_PATH
  ARENA_OPENAPI_BASE
  ARENA_OPENAPI_TOKEN
  ARENA_LLM_API_KEY
  SWE_RL_ADMIN_API_KEY
  WANDB_API_KEY
  WANDB_BASE_URL
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "${variable_name} must be set in ${ARENA_RUN_ENV_FILE}" >&2
    exit 1
  fi
done

if [[ "${1:-}" == "--check" ]]; then
  echo "Arena rollout environment is ready."
  echo "trial_name=${TRIAL_NAME}"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

unset ARENA_ROLLOUT_DATA_ID
mkdir -p "${AREAL_DIR}/rl_logs"

job_id=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/sbatch_arena_rollout_only.sh")
echo "Submitted Arena rollout-only job ${job_id}"
echo "trial_name=${TRIAL_NAME}"
echo "log=${AREAL_DIR}/rl_logs/swe-arena-rollout-only-${job_id}.out"
