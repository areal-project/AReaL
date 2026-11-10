#!/usr/bin/env bash
# Single-node example; override partition/account/GRES/CPU/RAM/time for your site.
#SBATCH --job-name=areal-tasks-sif
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${TASK_CODE_ROOT:?Absolute AReaL checkout}"
: "${TASK_MOUNT_ROOT:?Shared parent containing code, models, tasks, venv and results}"
: "${TASK_TRAIN_SIF:?Prepared training SIF}"
: "${TASK_IMAGE_MANIFEST:?Host-side image alias/SIF/hash manifest}"
: "${TASK_RUN_ROOT:?New absolute run directory}"
export TASK_MODE="${TASK_MODE:-train}" TASK_CASE="${TASK_CASE:-hermes-dapo}"
export TASK_PYTHON="${TASK_PYTHON:-python3}"
for name in TASK_CODE_ROOT TASK_MOUNT_ROOT TASK_TRAIN_SIF TASK_IMAGE_MANIFEST TASK_RUN_ROOT; do
  [[ "${!name}" == /* ]]
done
mkdir -p "$(dirname "$TASK_RUN_ROOT")"
mkdir "$TASK_RUN_ROOT"
export AREAL_TASK_RUNTIME=singularity
socket_dir=$(mktemp -d /tmp/areal-sif.XXXXXXXX)
export AREAL_TASK_HOST_SOCKET="$socket_dir/tools.sock"
host_pid=""
cleanup() {
  if [[ -n "$host_pid" ]]; then
    kill -TERM "$host_pid" 2>/dev/null || true
    wait "$host_pid" 2>/dev/null || true
  fi
  rm -rf -- "$socket_dir"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd "$TASK_CODE_ROOT"
host_python="${TASK_HOST_PYTHON:-python3}"
"$host_python" -S -c 'import sys; assert sys.version_info >= (3, 6, 8), "Host Python >=3.6.8 required"; print(sys.version)' > "$TASK_RUN_ROOT/host-python-version.txt"
singularity --version > "$TASK_RUN_ROOT/runtime-version.txt"
"$host_python" -S -m examples.v2_online_workflows.environments.singularity_host serve \
  --socket "$AREAL_TASK_HOST_SOCKET" --root "$TASK_RUN_ROOT/host-tools" \
  --images "$TASK_IMAGE_MANIFEST" --workers "${TASK_TOOL_WORKERS:-4}" \
  --overlay-mb "${TASK_OVERLAY_MB:-4096}" \
  --proc-mode "${TASK_PROC_MODE:-default}" \
  > "$TASK_RUN_ROOT/host-tools.log" 2>&1 &
host_pid=$!
printf '%s\n' "$host_pid" > "$TASK_RUN_ROOT/host-tools.pid"
ready=0
for ((attempt=0; attempt<1800; attempt++)); do
  kill -0 "$host_pid" 2>/dev/null || exit 1
  if [[ -S "$AREAL_TASK_HOST_SOCKET" ]]; then ready=1; break; fi
  sleep 1
done
[[ "$ready" == 1 ]]

# --cleanenv does not remove explicit APPTAINERENV_* injection. Pass only job inputs.
while IFS='=' read -r name value; do
  case "$name" in APPTAINERENV_*|SINGULARITYENV_*) unset "$name" ;; esac
done < <(env)
for name in TASK_CODE_ROOT TASK_RUN_ROOT TASK_MODE TASK_CASE TASK_PYTHON \
  TASK_TRAIN_TASKS TASK_EVAL_TASKS TASK_MODEL_PATH TASK_N TASK_EPOCHS TASK_BATCH \
  TASK_CONCURRENCY TASK_SAVE_EVERY TASK_TOOL_WORKERS TASK_HW SLURM_JOB_ID CUDA_VISIBLE_DEVICES \
  AREAL_TASK_RUNTIME AREAL_TASK_HOST_SOCKET AREAL_TASK_PYTHON_IMAGE \
  AREAL_SWE_EVALUATOR_ROOT AREAL_INFERENCE_ADMIN_KEY AREAL_AGENT_ADMIN_KEY AREAL_ACTOR_ADMIN_KEY; do
  if declare -p "$name" >/dev/null 2>&1; then export "APPTAINERENV_${name}=${!name}"; fi
done
export APPTAINERENV_PYTHONPATH="$TASK_CODE_ROOT"
container_command=(singularity exec --cleanenv --no-home)
if [[ "$TASK_MODE" != probe && "$TASK_MODE" != reference ]]; then container_command+=(--nv); fi
srun --mpi=none "${container_command[@]}" \
  --bind "$TASK_MOUNT_ROOT:$TASK_MOUNT_ROOT" --bind "$socket_dir:$socket_dir" \
  "$TASK_TRAIN_SIF" bash "$TASK_CODE_ROOT/examples/v2_online_workflows/singularity_job.sh"
