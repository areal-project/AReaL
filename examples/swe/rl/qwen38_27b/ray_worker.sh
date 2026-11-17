#!/usr/bin/env bash
# Run inside the user-provided Ray image, before importing any Python packages.
set -euo pipefail
ray_role=${1:?Expected controller, actor, or rollout}
shift
case "$ray_role" in
  controller|actor|rollout) ;;
  *) echo "Unknown Qwen Ray role: $ray_role" >&2; exit 2 ;;
esac
ray_recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${QWEN_ENV_FILE:?Set a shared Ray site profile}"
# These values are pinned by the controller, even if the profile uses defaults.
ray_profile=$QWEN_ENV_FILE
ray_trial=${QWEN_SUBMITTED_TRIAL:-}
ray_run_dir=${QWEN_SUBMITTED_RUN_DIR:-}
ray_experiment=${QWEN_SUBMITTED_EXPERIMENT:-}
ray_output=${QWEN_SUBMITTED_OUTPUT_ROOT:-}
ray_user=${QWEN_SUBMITTED_USER:-}
ray_python=${QWEN_RAY_PYTHON:-}
# Preserve the device/rank assignment made by Ray, including grouped colocation.
ray_device_vars=(CUDA_VISIBLE_DEVICES SLURM_LOCALID SLURM_NODEID SLURM_NNODES)
declare -A ray_devices=()
for ray_key in "${ray_device_vars[@]}"; do
  if [[ -v $ray_key ]]; then ray_devices[$ray_key]=${!ray_key}; fi
done
export QWEN_RUNTIME_ROLE=$ray_role
set -a
source "$ray_profile"
set +a
for ray_key in "${ray_device_vars[@]}"; do
  if [[ ${ray_devices[$ray_key]+present} ]]; then
    export "$ray_key=${ray_devices[$ray_key]}"
  else
    unset "$ray_key"
  fi
done
export QWEN_ENV_FILE=$ray_profile
export QWEN_REPO=$(cd "$ray_recipe/../../../.." && pwd)
export QWEN_RECIPE_DIR=$ray_recipe
export QWEN_RUNTIME_ROLE=$ray_role
export QWEN_ARENA_TRIAL=${ray_trial:-${QWEN_ARENA_TRIAL:?Set a unique trial}}
export QWEN_EXPERIMENT_NAME=${ray_experiment:-${QWEN_EXPERIMENT_NAME:-qwen38-27b-arena-rl}}
export QWEN_OUTPUT_ROOT=${ray_output:-${QWEN_OUTPUT_ROOT:?Set shared output storage}}
export QWEN_RUN_USER=${ray_user:-${QWEN_RUN_USER:-${SUDO_USER:-$(id -un)}}}
export QWEN_RUN_DIR=${ray_run_dir:-${QWEN_RUN_DIR:?Set a shared run directory}}
export QWEN_RAY_PYTHON=${ray_python:-${QWEN_RAY_PYTHON:-python3}}
# A Slurm allocation may host Ray, but its global task rank is not this RPC
# worker's index. Keep the LOCALID/NODEID/NNODES assigned for grouped AWEX.
unset SLURM_PROCID RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT AREAL_SPMD_MODE
export QWEN_ARENA_OUTPUT=$QWEN_OUTPUT_ROOT
export USER=$QWEN_RUN_USER LOGNAME=$QWEN_RUN_USER
for ray_key in QWEN_MODEL ARENA_OPENAPI_BASE ARENA_OPENAPI_TOKEN ARENA_LLM_API_KEY WANDB_BASE_URL WANDB_API_KEY; do
  : "${!ray_key:?Set $ray_key in the Ray site profile}"
  export "$ray_key"
done
test -f "$QWEN_MODEL/config.json"
# Do not inherit the controller's site-packages in rollout. Images provide the
# packages; optional per-role paths support separately installed dependency trees.
case "$ray_role" in
  rollout)
    export PYTHONPATH="${QWEN_RAY_ROLLOUT_PYTHONPATH:+$QWEN_RAY_ROLLOUT_PYTHONPATH:}$QWEN_REPO:$ray_recipe"
    export AWEX_ACTOR_ALLOC_CONF=expandable_segments:False
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    export SGLANG_VLM_CACHE_SIZE_MB=100
    export SGLANG_NUMA_BIND_V2=0
    ;;
  actor|controller)
    export PYTHONPATH="${QWEN_RAY_ACTOR_PYTHONPATH:+$QWEN_RAY_ACTOR_PYTHONPATH:}$QWEN_REPO:$ray_recipe"
    export AWEX_ACTOR_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
    ;;
esac
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=areal.engine.sglang_processors
export SGLANG_VLM_QWEN_LIMIT_PIXEL=${SGLANG_VLM_QWEN_LIMIT_PIXEL:-true}
export AREAL_VLM_ALIGN_SGLANG=${AREAL_VLM_ALIGN_SGLANG:-true}
export TMS_INIT_ENABLE=0 TMS_INIT_ENABLE_CPU_BACKUP=0
export QWEN_PLE_CHUNK_TOKENS=0 QWEN_GDN_CP_COMPAT=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NO_PROXY='*' no_proxy='*'
unset QWEN_AWEX_FROZEN_CONTRACT MCORE_BRIDGE_ROOT
unset QWEN_QSA_STABLE_TOPK QWEN_ISOLATE_REQUEST_CACHE NCCL_ALGO NCCL_PROTO CUBLAS_WORKSPACE_CONFIG
unset QWEN_BATCH_REPLAY_PATH QWEN_BATCH_REPLAY_PATHS QWEN_BATCH_SNAPSHOT_DIR
unset QWEN_ARENA_TASK_IDS_FILE QWEN_ARENA_STREAMS_FILE
ray_process_id="$(hostname)-$ray_role-$$"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/areal-q27-$QWEN_ARENA_TRIAL-$ray_process_id/cache"
mkdir -p "$XDG_CACHE_HOME"
cd "$QWEN_REPO"
case "$ray_role" in
  controller)
    exec "$QWEN_RAY_PYTHON" -m examples.swe.rl.qwen38_27b.ray_train "$@"
    ;;
  actor)
    exec "$QWEN_RAY_PYTHON" -m qwen27_actor_worker "$@"
    ;;
  rollout)
    export SGLANG_LOGGING_CONFIG_PATH="$QWEN_OUTPUT_ROOT/logs/$QWEN_RUN_USER/$QWEN_EXPERIMENT_NAME/$QWEN_ARENA_TRIAL/native/sglang-$ray_process_id.json"
    "$QWEN_RAY_PYTHON" "$ray_recipe/configure_sglang_logging.py"
    exec "$QWEN_RAY_PYTHON" -m areal.infra.rpc.rpc_server "$@"
    ;;
esac
