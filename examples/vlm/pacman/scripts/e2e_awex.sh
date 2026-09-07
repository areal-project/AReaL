#!/usr/bin/env bash
# GPU environment only; not run locally. Eight colocated GPUs and both curricula.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
: "${PACMAN_ARTIFACT_ROOT:?Use a fresh absolute output directory}"
cd "$AREAL_ROOT"
export AREAL_SPMD_MODE=0
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
bash examples/vlm/pacman/scripts/train_c1_awex.sh \
    total_train_steps="${PACMAN_TRAIN_STEPS:-2}" "$@"
CURRICULUM1_CHECKPOINT=$(python -m examples.vlm.pacman.checkpoint "$PACMAN_ARTIFACT_ROOT/curriculum1/training")
export CURRICULUM1_CHECKPOINT
python -m examples.vlm.pacman.train \
    --config examples/vlm/pacman/curriculum2_awex_colocate.yaml \
    total_train_steps="${PACMAN_TRAIN_STEPS:-2}" "$@"
python -m examples.vlm.pacman.checkpoint "$PACMAN_ARTIFACT_ROOT/curriculum2/training"
