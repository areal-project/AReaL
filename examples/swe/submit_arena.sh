#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Submit one CPU controller; AReaL schedules the GPU workers from CONFIG_PATH.
set -euo pipefail
set +x

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: bash examples/swe/submit_arena.sh [--check-arena]

Required environment:
  AREAL_IMAGE, AREAL_PYTHON, MODEL_PATH, AREAL_FILEROOT,
  ARENA_CREDENTIALS_FILE, AREAL_CONTROLLER_MOUNTS, AREAL_WORKER_MOUNTS
Optional:
  AREAL_DIR (script's checkout), CONFIG_PATH (examples/swe/arena_grpo.yaml),
  TRIAL_NAME (unique timestamp), AREAL_CONTAINER_BIN (apptainer),
  SBATCH_PARTITION, SBATCH_ACCOUNT, SBATCH_RESERVATION, SBATCH_TIMELIMIT,
  ARENA_CONTROLLER_TIME (01:30:00), ARENA_PREFLIGHT_MIN_TASKS (1).

Use CONFIG_PATH to choose the single- or multi-stream YAML and GPU topology.
--check-arena submits a CPU-only validation job, without starting GPU workers.
Credentials stay in the shared file; never supply tokens as command arguments.
EOF
    exit 0 ;;
  ""|--check-arena|--controller|--run) ;;
  *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected at most one argument' >&2; exit 2; }

# sbatch copies this script to its spool: resolve the checkout BEFORE submission.
if [[ ${1:-} != --controller && ${1:-} != --run ]]; then
  export AREAL_DIR=${AREAL_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}
  export CONFIG_PATH=${CONFIG_PATH:-examples/swe/arena_grpo.yaml}
  export TRIAL_NAME=${TRIAL_NAME:-arena-$(date +%Y%m%d-%H%M%S)-$$}
  export ARENA_CHECK_ONLY=${1:-}
fi
: "${AREAL_DIR:?Set AREAL_DIR to the shared checkout}"
: "${CONFIG_PATH:?Set CONFIG_PATH to the training YAML}"
: "${AREAL_IMAGE:?Set AREAL_IMAGE to the container image}"
: "${AREAL_PYTHON:?Set AREAL_PYTHON to Python inside the container}"
: "${MODEL_PATH:?Set MODEL_PATH to the shared model directory}"
: "${AREAL_FILEROOT:?Set AREAL_FILEROOT to shared experiment storage}"
: "${ARENA_CREDENTIALS_FILE:?Set ARENA_CREDENTIALS_FILE to a shared private env file}"
: "${AREAL_CONTROLLER_MOUNTS:?Set container binds including shared storage and Slurm access}"
: "${AREAL_WORKER_MOUNTS:?Set worker container binds for code, model, credentials and output}"
export AREAL_DIR CONFIG_PATH AREAL_IMAGE AREAL_PYTHON MODEL_PATH AREAL_FILEROOT
export ARENA_CREDENTIALS_FILE AREAL_CONTROLLER_MOUNTS AREAL_WORKER_MOUNTS
export TRIAL_NAME SLURM_EXPORT_ENV=ALL
export SBATCH_TIMELIMIT=${SBATCH_TIMELIMIT:-01:00:00}
cd -- "$AREAL_DIR"
for name in AREAL_DIR AREAL_IMAGE MODEL_PATH AREAL_FILEROOT ARENA_CREDENTIALS_FILE; do
  [[ ${!name} == /* ]] || { echo "$name must be an absolute shared path" >&2; exit 2; }
done
[[ -r $CONFIG_PATH && -r $AREAL_IMAGE && -d $MODEL_PATH && -r $ARENA_CREDENTIALS_FILE ]] || {
  echo 'Config, image, model or credentials file is missing/unreadable' >&2; exit 2;
}
export ARENA_SUBMISSION_DIR="$AREAL_FILEROOT/submissions/$TRIAL_NAME"
mkdir -p -- "$ARENA_SUBMISSION_DIR"

if [[ ${1:-} == --controller ]]; then
  # Site-specific Slurm/munge mounts belong in the environment, not this script.
  exec "${AREAL_CONTAINER_BIN:-apptainer}" exec --no-eval --pid --writable-tmpfs \
    --bind "$AREAL_CONTROLLER_MOUNTS" "$AREAL_IMAGE" \
    bash "$AREAL_DIR/examples/swe/submit_arena.sh" --run
elif [[ ${1:-} == --run ]]; then
  # Older credential files also set deployment options. Explicit launch values win.
  declare -A launch_env=()
  for name in AREAL_DIR CONFIG_PATH AREAL_IMAGE AREAL_PYTHON MODEL_PATH AREAL_FILEROOT \
    ARENA_CREDENTIALS_FILE AREAL_WORKER_MOUNTS TRIAL_NAME ARENA_SUBMISSION_DIR ARENA_CHECK_ONLY \
    ARENA_STREAM_ID ARENA_OPENAPI_BASE ARENA_PREFLIGHT_MIN_TASKS SBATCH_TIMELIMIT; do
    if [[ -v $name ]]; then
      launch_env[$name]=${!name}
    fi
  done
  set -a
  source "$ARENA_CREDENTIALS_FILE"
  set +a
  for name in "${!launch_env[@]}"; do
    export "$name=${launch_env[$name]}"
  done
  : "${ARENA_OPENAPI_BASE:?Credentials must export ARENA_OPENAPI_BASE}"
  : "${ARENA_OPENAPI_TOKEN:?Credentials must export ARENA_OPENAPI_TOKEN}"
  : "${ARENA_LLM_API_KEY:?Credentials must export ARENA_LLM_API_KEY}"
  : "${SWE_RL_ADMIN_API_KEY:?Credentials must export SWE_RL_ADMIN_API_KEY}"
  export SLURM_EXPORT_ENV=ALL
  unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  export PYTHONPATH="$AREAL_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
  export PATH="$(dirname -- "$AREAL_PYTHON"):$PATH"
  export APPTAINER_BIND="$AREAL_WORKER_MOUNTS" SINGULARITY_BIND="$AREAL_WORKER_MOUNTS"
  # Resolve exactly the same config used by training, including Hydra defaults.
  "$AREAL_PYTHON" - "$CONFIG_PATH" <<'PY'
import base64
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from examples.swe.arena_config import load_arena_stream_configs
from examples.swe.arena_preflight import validate_streams
from examples.swe.utils import SWEPPOConfig
from areal.api.cli_args import load_expr_config

args = ["--config", sys.argv[1]]
config, _ = load_expr_config(args, SWEPPOConfig)
if config.scheduler.type != "slurm" or config.econfig.dataset_source != "arena":
    raise ValueError("This entrypoint requires the Slurm scheduler and Arena dataset")
streams = load_arena_stream_configs(config.econfig)
payload = yaml.safe_dump({"streams": [asdict(stream) for stream in streams]})
summary = validate_streams(
    streams_yaml_b64=base64.b64encode(payload.encode()).decode(),
    base_url=config.econfig.arena_base_url,
    min_terminal_tasks=int(os.environ.get("ARENA_PREFLIGHT_MIN_TASKS", "1")),
)
Path(os.environ["ARENA_SUBMISSION_DIR"], "preflight.json").write_text(
    json.dumps({"streams": summary}, indent=2) + "\n", encoding="utf-8"
)
if not os.environ.get("ARENA_CHECK_ONLY"):
    os.execv(sys.executable, [sys.executable, "-m", "examples.swe.train_swe_rl", *args])
PY
else
  exec sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
    --gres=gpu:0 --time="${ARENA_CONTROLLER_TIME:-01:30:00}" --export=ALL \
    --job-name="$TRIAL_NAME-controller" --chdir="$AREAL_DIR" \
    --output="$ARENA_SUBMISSION_DIR/controller-%j.log" \
    "$AREAL_DIR/examples/swe/submit_arena.sh" --controller
fi
