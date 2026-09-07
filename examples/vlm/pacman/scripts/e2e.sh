#!/usr/bin/env bash
# GPU environment only; not run locally. Eight GPUs and both real curricula.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
: "${MAAPACMAN_PACMAN_ROOT:?Export the pinned pacman-python checkout}"
: "${PACMAN_ARTIFACT_ROOT:?Use a fresh absolute output directory}"
cd "$AREAL_ROOT"
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
# Default: two updates per stage. Set PACMAN_TRAIN_STEPS=null for 5 x 20 updates.
python -m areal.infra.launcher.local examples/vlm/pacman/train.py \
    --config examples/vlm/pacman/curriculum1.yaml \
    total_train_steps="${PACMAN_TRAIN_STEPS:-2}" "$@"
CURRICULUM1_CHECKPOINT=$(python -m examples.vlm.pacman.checkpoint "$PACMAN_ARTIFACT_ROOT/curriculum1/training")
export CURRICULUM1_CHECKPOINT
python -m areal.infra.launcher.local examples/vlm/pacman/train.py \
    --config examples/vlm/pacman/curriculum2.yaml \
    total_train_steps="${PACMAN_TRAIN_STEPS:-2}" "$@"
python -m examples.vlm.pacman.checkpoint "$PACMAN_ARTIFACT_ROOT/curriculum2/training"
