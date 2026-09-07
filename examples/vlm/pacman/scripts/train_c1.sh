#!/usr/bin/env bash
# GPU environment only; not run locally. C1 with the eight-GPU release template.
# Default: Qwen3.5-9B, 512 environment steps, 12 episodes/state, 100 updates.
# Example smoke override: bash examples/vlm/pacman/scripts/train_c1.sh total_train_steps=2
set -euo pipefail

: "${AREAL_ROOT:?Export the current AReaL checkout containing the Pacman plugins}"
: "${MAAPACMAN_PACMAN_ROOT:?Export the pinned pacman-python checkout}"
: "${PACMAN_ARTIFACT_ROOT:?Use a fresh absolute output directory}"

cd "$AREAL_ROOT"
export AREAL_SPMD_MODE=0
export SGLANG_RETURN_ORIGINAL_LOGPROB=0

# Activate the GPU environment and install the pinned areal-pacman package first.
# Start one controller; scheduler.type=local in the recipe creates the workers.
# Additional arguments use AReaL's key=value configuration overrides.
exec python -m examples.vlm.pacman.train \
    --config examples/vlm/pacman/curriculum1.yaml \
    "$@"
