#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Flash V3 Arena profiles from swe-dev; deployment paths come from the environment.
set -euo pipefail
set +x

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash examples/swe/submit_arena_flash_v3.sh [--print-profile|--check-arena]

N_NODES selects 2, 4, 8 (default), or 16 shared training/inference worker nodes.
Defaults: batch 32, 12 samples, 128K context, 500 steps, two Arena streams.
Supply submit_arena.sh's required environment plus ROLLOUT_IMAGE, AWEX_ROOT,
FLASH_LINEAR_ATTENTION_ROOT. MODEL_PATH selects a Flash V3 checkpoint.
ACTOR_RUNTIME_PYTHONPATH and ROLLOUT_RUNTIME_PYTHONPATH select optional overlays.
DATASET_CONFIG overrides the versioned streams YAML. See README_arena.md for
runtime requirements and differences from the complete swe-dev recipe.
HELP
    exit 0 ;;
  ""|--print-profile|--check-arena) ;;
  *) echo 'Unknown argument; use --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected at most one argument' >&2; exit 2; }
export N_NODES=${N_NODES:-8}
case "$N_NODES" in
  2) actor='(attn:d1p2t4c2|ffn:d1p2e8)'; rollout=d4t4p1 ;;
  4) actor='(attn:d2p2t4c2|ffn:d2p2e8)'; rollout=d8t4p1 ;;
  8) actor='(attn:d2p2t2c8|ffn:d4p2e8)'; rollout=d16t4p1 ;;
  16) actor='(attn:d4p2t2c8|ffn:d8p2e8)'; rollout=d32t4p1 ;;
  *) echo 'N_NODES must be 2, 4, 8 or 16' >&2; exit 2 ;;
esac
export ACTOR_BACKEND="megatron:$actor" ROLLOUT_BACKEND="sglang:$rollout"
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32} N_SAMPLES=${N_SAMPLES:-12}
export TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-500}
export MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-1500}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
[[ $CONTEXT_LENGTH =~ ^[1-9][0-9]*$ ]] && (( CONTEXT_LENGTH >= 2 )) || {
  echo 'CONTEXT_LENGTH must be an integer of at least 2' >&2; exit 2;
}
export MAX_TOKENS=${MAX_TOKENS:-$((CONTEXT_LENGTH - 1))}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-$MAX_TOKENS}
for name in MAX_TOKENS MAX_NEW_TOKENS; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] && (( ${!name} < CONTEXT_LENGTH )) || {
    echo "$name must be positive and smaller than CONTEXT_LENGTH" >&2; exit 2;
  }
done
export ACTOR_MAX_TOKENS_PER_MB=${ACTOR_MAX_TOKENS_PER_MB:-$CONTEXT_LENGTH}
if [[ ${1:-} == --print-profile ]]; then
  printf '%s\n' "nodes=$N_NODES" "actor=$ACTOR_BACKEND" "rollout=$ROLLOUT_BACKEND" \
    "context=$CONTEXT_LENGTH" "max_tokens=$MAX_TOKENS" "max_new_tokens=$MAX_NEW_TOKENS" \
    "batch=$TRAIN_BATCH_SIZE" "samples=$N_SAMPLES" \
    "steps=$TOTAL_TRAIN_STEPS" "concurrency=$MAX_CONCURRENT_ROLLOUTS"
  exit 0
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export AREAL_DIR=${AREAL_DIR:-$(cd -- "$script_dir/../.." && pwd -P)}
export CONFIG_PATH=${CONFIG_PATH:-$AREAL_DIR/examples/swe/arena_flash_v3.yaml}
export DATASET_CONFIG=${DATASET_CONFIG:-$AREAL_DIR/examples/swe/dataset_configs/flash-v3-multi-stream.yaml}
: "${ROLLOUT_IMAGE:?Set ROLLOUT_IMAGE to the Flash V3 SGLang runtime}"
: "${AWEX_ROOT:?Set AWEX_ROOT to an AWEX checkout with the Flash V3 converter}"
: "${FLASH_LINEAR_ATTENTION_ROOT:?Set FLASH_LINEAR_ATTENTION_ROOT to compatible FLA sources}"
export ROLLOUT_IMAGE AWEX_ROOT FLASH_LINEAR_ATTENTION_ROOT
[[ -r $ROLLOUT_IMAGE && -r $DATASET_CONFIG && -d $AWEX_ROOT && -d $FLASH_LINEAR_ATTENTION_ROOT ]] || {
  echo 'Rollout image, streams YAML, AWEX or FLA sources are missing/unreadable' >&2; exit 2;
}
exec bash "$AREAL_DIR/examples/swe/submit_arena.sh" "$@"
