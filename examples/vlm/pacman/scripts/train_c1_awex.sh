#!/usr/bin/env bash
# Eight colocated GPUs: Megatron DP=4/TP=2 and eight SGLang TP=1 instances.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
exec bash "$AREAL_ROOT/examples/vlm/pacman/scripts/train_c1.sh" \
    --config examples/vlm/pacman/curriculum1_awex_colocate.yaml "$@"
