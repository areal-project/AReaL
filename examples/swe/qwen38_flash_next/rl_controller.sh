#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
profile=${1:?Expected swe or swe-eval}; shift
recipe_dir="$QWEN_REPO/examples/swe/qwen38_flash_next"
export NO_PROXY='*' no_proxy='*'
export PYTHONPATH=$QWEN_CONTROLLER_PYTHONPATH
cd "$QWEN_REPO"
case "$profile" in
  swe|swe-eval)
    set -a; source "$QWEN_PRIVATE_ENV"; set +a
    : "${ARENA_OPENAPI_BASE:?Set Arena endpoint in private environment}"
    : "${ARENA_LLM_API_KEY:?Set Arena LLM gateway credentials in private environment}"
    export ARENA_OPENAPI_BASE ARENA_LLM_API_KEY
    export QWEN_ARENA_TRIAL=${QWEN_ARENA_TRIAL:-qwen_flash_next_$SLURM_JOB_ID}
    export QWEN_ARENA_OUTPUT="$QWEN_OUTPUT_ROOT/$QWEN_ARENA_TRIAL"
    export SWE_RL_ADMIN_API_KEY
    SWE_RL_ADMIN_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    mkdir -p "$QWEN_ARENA_OUTPUT"
    config=${QWEN_CONFIG:-$recipe_dir/swe_mm_rl.yaml}
    if [[ $profile == swe-eval ]]; then
      : "${QWEN_ARENA_TASK_IDS_FILE:?Set the exact reference task manifest}"
      test -f "$QWEN_ARENA_TASK_IDS_FILE"
    fi
    exec python3 -m examples.swe.qwen38_flash_next.train_rl "$profile" \
      --config "$config" "$@"
    ;;
  *) echo 'Expected swe or swe-eval' >&2; exit 2 ;;
esac
