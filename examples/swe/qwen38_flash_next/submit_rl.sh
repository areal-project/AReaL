#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: bash submit_rl.sh swe|swe-eval [training config overrides...]
# Inference: SGLang 0.5.19.dev125+g119b5ffe4 in a fresh writable container layer.
# Both QSA patches run before rollout startup and reject other/already-patched sources.
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export QWEN_REPO=${QWEN_REPO:-$(cd "$recipe_dir/../../.." && pwd)}
profile=${1:?Usage: submit_rl.sh swe|swe-eval [overrides...]}
shift
case "$profile" in swe|swe-eval) ;; *) echo 'Expected swe or swe-eval' >&2; exit 2 ;; esac
if [[ -n ${QWEN_LAUNCH_ENV:-} ]]; then
  set -a; source "$QWEN_LAUNCH_ENV"; set +a
fi
export QWEN_OUTPUT_ROOT=${QWEN_OUTPUT_ROOT:-${QWEN_EXPERIMENTS_ROOT:-/storage/openpsi/experiments}/qwen38-flash-next/$profile}
for name in QWEN_OUTPUT_ROOT QWEN_MODEL QWEN_ACTOR_IMAGE QWEN_ROLLOUT_IMAGE \
  QWEN_RESERVATION QWEN_NODELIST QWEN_PARTITION QWEN_CONTROLLER_NODE \
  QWEN_MOUNTS QWEN_CONTROLLER_MOUNTS MEGATRON_ROOT; do
  : "${!name:?Set $name in the launch environment}"
  export "$name"
done
for name in QWEN_PRIVATE_ENV QWEN_ARENA_STREAMS_FILE; do
  : "${!name:?Set $name for SWE}"
  test -f "${!name}"
  export "$name"
done
# A fresh shared environment per submission cannot change an already-running job.
[[ $QWEN_OUTPUT_ROOT = /* ]] || { echo 'QWEN_OUTPUT_ROOT must be absolute' >&2; exit 1; }
mkdir -p "$QWEN_OUTPUT_ROOT"
qwen_runtime=$(mktemp -d "$QWEN_OUTPUT_ROOT/uv-runtime-XXXXXXXX")
export QWEN_TRAIN_PYTHON="$qwen_runtime/bin/python"
singularity exec --no-eval --pid --writable-tmpfs \
  --bind "$QWEN_CONTROLLER_MOUNTS" \
  "$QWEN_ACTOR_IMAGE" bash -s -- "$QWEN_REPO" "$qwen_runtime" <<'PREPARE'
set -euo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
cd "$1"
# Retain CUDA extensions supplied by the actor image; uv installs locked packages
# in the new environment without modifying the image's site-packages.
uv venv --python python3 --system-site-packages "$2"
UV_PROJECT_ENVIRONMENT="$2" uv sync --locked --extra cuda \
  --no-group transformers-default --group qwen-flash-next
"$2/bin/python" - <<'PYTHON'
import importlib
import importlib.metadata as metadata
import json

if metadata.version("transformers") != "5.16.1":
    raise RuntimeError("Expected Transformers 5.16.1 in the prepared runtime")
bridge = metadata.distribution("mcore-bridge")
origin = json.loads(bridge.read_text("direct_url.json"))
if origin["vcs_info"]["commit_id"] != "557aaf93b16d083fdec4f82a8251d47d47c76ccb":
    raise RuntimeError("Prepared runtime has the wrong mcore-bridge commit")
for module in ("transformers.models.qwen4_exp.modeling_qwen4_exp", "mcore_bridge",
               "megatron.core", "transformer_engine.pytorch", "flash_attn"):
    importlib.import_module(module)
PYTHON
PREPARE
# Check the worker mount view as well as the controller's before scheduling.
singularity exec --no-eval --pid --writable-tmpfs \
  --bind "$QWEN_MOUNTS" \
  "$QWEN_ACTOR_IMAGE" bash -c 'test -x "$1" && test -d "$2"' \
  bash "$QWEN_TRAIN_PYTHON" "$QWEN_REPO"
export QWEN_ACTOR_PYTHONPATH="$QWEN_REPO"
export QWEN_ROLLOUT_PYTHONPATH="${QWEN_INFER_EXTRA_PYTHONPATH:+$QWEN_INFER_EXTRA_PYTHONPATH:}$MEGATRON_ROOT:$QWEN_REPO"
export QWEN_CONTROLLER_PYTHONPATH=$QWEN_ACTOR_PYTHONPATH
export SBATCH_PARTITION=$QWEN_PARTITION SBATCH_RESERVATION=$QWEN_RESERVATION
exec sbatch --partition="$QWEN_PARTITION" --reservation="$QWEN_RESERVATION" \
  --nodelist="$QWEN_CONTROLLER_NODE" --chdir="$QWEN_REPO" \
  --job-name="qwen38-$profile-controller" --output="$QWEN_OUTPUT_ROOT/controller-%j.log" \
  --export=ALL "$recipe_dir/rl_controller.sbatch" "$profile" "$@"
