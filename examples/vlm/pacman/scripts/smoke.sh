#!/usr/bin/env bash
# GPU environment only; not run locally. Two GPUs, a small VLM, one update.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
: "${MAAPACMAN_PACMAN_ROOT:?Export the pinned pacman-python checkout}"
: "${PACMAN_ARTIFACT_ROOT:?Use a fresh absolute output directory}"
cd "$AREAL_ROOT"
export AREAL_SPMD_MODE=0
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
python -m pytest tests/test_rl_plugins.py examples/vlm/pacman/tests -m 'not slow'
python -m pytest examples/vlm/pacman/tests/test_policy_tp.py -m slow
python -m examples.vlm.pacman.train \
    --config examples/vlm/pacman/curriculum1.yaml \
    actor.path="${PACMAN_SMOKE_MODEL:-Qwen/Qwen3.5-0.8B}" \
    cluster.n_gpus_per_node=2 actor.backend=megatron:d1p1t1 rollout.backend=sglang:d1p1t1 \
    train_dataset.batch_size=1 valid_dataset.batch_size=1 \
    dataset_generation.train_episodes=1 dataset_generation.validation_episodes=1 \
    environment.max_steps=32 planner_audit.max_steps=32 total_train_steps=1 \
    "$@"
python -m examples.vlm.pacman.checkpoint "$PACMAN_ARTIFACT_ROOT/curriculum1/training"
