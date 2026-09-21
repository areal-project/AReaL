#!/usr/bin/env bash
# Launch OSWorld GRPO training against a vendor-neutral remote sandbox
# cluster behind an HTTPS gateway.
#
# Usage:
#   bash examples/osworld/run_train.sh [smoke|full] [extra hydra-style overrides...]
#
# Examples:
#   # smoke run with the defaults below
#   bash examples/osworld/run_train.sh smoke
#
#   # full run with bumped concurrency
#   bash examples/osworld/run_train.sh full rollout.max_concurrent_rollouts=4 n_trajs=2
#
#   # override base model on the fly
#   bash examples/osworld/run_train.sh smoke actor.path=/path/to/other/model
#
# Required env vars:
#   OSWORLD_SANDBOX_TOKEN     — application secret for the gateway. Don't commit this.
#   OSWORLD_SANDBOX_ENDPOINT  — gateway URL (no default; must be set explicitly).
#
# Optional env vars:
#   AREAL_TEXT_ONLY_MODEL     — path to a text-only base model; required for the
#                               `smoke-text` stage. No default.
#   AREAL_ENV_PREFIX          — conda env prefix (default: ../../../env)
#   AREAL_REPO                — AReaL checkout root (default: parent of this script's grandparent)
#   CONDA_PREFIX_BASE         — base conda install (default: $HOME/conda); used to
#                               source `etc/profile.d/conda.sh`. Override if conda
#                               lives elsewhere (e.g. /opt/conda).
#   NO_PROXY                  — pre-existing no-proxy list; this script appends its
#                               own generic CIDRs. Append your own internal domains
#                               here before invocation if needed.
#   STAGE                     — alternative to first positional arg

set -euo pipefail

# -------- locate repo + env --------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_REPO="${AREAL_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
AREAL_ENV_PREFIX="${AREAL_ENV_PREFIX:-${AREAL_REPO}/../env}"

if [[ ! -d "${AREAL_ENV_PREFIX}" ]]; then
    echo "[run_train.sh] conda env not found at: ${AREAL_ENV_PREFIX}" >&2
    echo "  set AREAL_ENV_PREFIX or rebuild via SETUP.md" >&2
    exit 1
fi

# Activate conda env (we need the real `conda activate` shell function).
# Conda's binutils activation script touches unbound vars, so relax `nounset`
# across the activate call. Override CONDA_PREFIX_BASE if conda lives outside
# $HOME/conda (e.g. CONDA_PREFIX_BASE=/opt/conda).
# shellcheck disable=SC1091
source "${CONDA_PREFIX_BASE:-$HOME/conda}/etc/profile.d/conda.sh"
set +u
conda activate "${AREAL_ENV_PREFIX}"
set -u

# Disable Python stdout/stderr block buffering so SGLang server logs reach
# the trial's rollout/0.log in real time — otherwise a setup-timeout SIGTERM
# truncates the buffer and we never see why SGLang failed to come up.
export PYTHONUNBUFFERED=1

# A corporate HTTP proxy (HTTP_PROXY / HTTPS_PROXY) cannot reach internal
# addresses, but the trainer's `_wait_for_server` uses `requests`, which
# honors those env vars by default. Without NO_PROXY the health probe to the
# local SGLang server (33.x / 10.x / localhost) is routed through the proxy
# and gets ECONNREFUSED, eventually triggering the 900s setup_timeout SIGTERM.
# Append our hosts/CIDRs to NO_PROXY (don't clobber an existing one). If you
# need to bypass the proxy for additional internal domains, set NO_PROXY to
# include them before invoking this script.
_areal_no_proxy_extra="localhost,127.0.0.1,0.0.0.0,33.0.0.0/8,10.0.0.0/8"
if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${NO_PROXY},${_areal_no_proxy_extra}"
else
    export NO_PROXY="${_areal_no_proxy_extra}"
fi
export no_proxy="${NO_PROXY}"

# -------- pick stage + token --------

STAGE="${1:-${STAGE:-smoke}}"
shift || true

if [[ "${STAGE}" != "smoke" && "${STAGE}" != "full" && "${STAGE}" != "smoke-text" ]]; then
    echo "[run_train.sh] first arg must be 'smoke', 'smoke-text', or 'full', got '${STAGE}'" >&2
    exit 2
fi

if [[ -z "${OSWORLD_SANDBOX_ENDPOINT:-}" ]]; then
    echo "[run_train.sh] OSWORLD_SANDBOX_ENDPOINT must be set in env" >&2
    echo "  e.g.  export OSWORLD_SANDBOX_ENDPOINT=https://your-gateway.example.com/path" >&2
    exit 3
fi
if [[ -z "${OSWORLD_SANDBOX_TOKEN:-}" ]]; then
    echo "[run_train.sh] OSWORLD_SANDBOX_TOKEN must be set in env" >&2
    echo "  e.g.  export OSWORLD_SANDBOX_TOKEN=sk-..." >&2
    exit 3
fi

# Auto-derive a trial name unless caller pinned one through extra args.
TRIAL_NAME="${STAGE}-$(date +%Y%m%d-%H%M%S)"

# -------- build override list --------

# Common overrides (every stage gets them).
COMMON_ARGS=(
    "gateway_endpoint=${OSWORLD_SANDBOX_ENDPOINT}"
    "gateway_token=${OSWORLD_SANDBOX_TOKEN}"
    "env_reset_wait_secs=30"
    "trial_name=${TRIAL_NAME}"
)

# Stage-specific defaults; extra positional args override these.
# AREAL_TEXT_ONLY_MODEL must be set by the user to run the `smoke-text` stage
# (no default — set it to a local HF checkpoint path).
_TEXT_ONLY_MODEL="${AREAL_TEXT_ONLY_MODEL:-}"

case "${STAGE}" in
    smoke)
        STAGE_ARGS=(
            "experiment_name=osworld-grpo-smoke"
            "rollout.max_concurrent_rollouts=1"
            "n_trajs=1"
            "max_steps=3"
            "train_dataset.batch_size=1"
            "total_train_epochs=1"
        )
        ;;
    smoke-text)
        # Plumbing smoke against a text-only base model. Strips screenshots
        # from the workflow so we don't need the VL training path
        # (mm_token_type_ids / multi_modal_input). Agent operates blind;
        # this is only useful for verifying the full PPO loop end-to-end.
        if [[ -z "${_TEXT_ONLY_MODEL}" ]]; then
            echo "[run_train.sh] stage 'smoke-text' requires AREAL_TEXT_ONLY_MODEL to be set" >&2
            echo "  e.g.  export AREAL_TEXT_ONLY_MODEL=/path/to/Qwen3-4B-Instruct-2507" >&2
            exit 4
        fi
        STAGE_ARGS=(
            "experiment_name=osworld-grpo-smoke-text"
            "rollout.max_concurrent_rollouts=1"
            "n_trajs=1"
            "max_steps=3"
            "train_dataset.batch_size=1"
            "total_train_epochs=1"
            "text_only=true"
            "actor.path=${_TEXT_ONLY_MODEL}"
            "tokenizer_path=${_TEXT_ONLY_MODEL}"
            "sglang.enable_multimodal=false"
        )
        ;;
    full)
        STAGE_ARGS=(
            "experiment_name=osworld-grpo"
            "rollout.max_concurrent_rollouts=2"
            "n_trajs=1"
            "max_steps=15"
            "train_dataset.batch_size=2"
        )
        ;;
esac

# -------- launch --------

cd "${AREAL_REPO}"

CONFIG_PATH="examples/osworld/config_osworld_sglang.yaml"

echo "[run_train.sh] stage=${STAGE} trial_name=${TRIAL_NAME}"
echo "[run_train.sh] env=${AREAL_ENV_PREFIX}"
echo "[run_train.sh] python=$(which python)"
echo "[run_train.sh] config=${CONFIG_PATH}"
echo "[run_train.sh] overrides:" "${COMMON_ARGS[@]}" "${STAGE_ARGS[@]}" "$@"

exec python -m examples.osworld.train \
    --config "${CONFIG_PATH}" \
    "${COMMON_ARGS[@]}" \
    "${STAGE_ARGS[@]}" \
    "$@"
