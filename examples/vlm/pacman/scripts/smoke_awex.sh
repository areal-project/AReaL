#!/usr/bin/env bash
# GPU environment only; not run locally. Two GPUs for TP checks, one for colocated RL.
# Two short updates exercise weight publication and the next rollout with new weights.
set -euo pipefail
: "${AREAL_ROOT:?Export the current AReaL checkout}"
cd "$AREAL_ROOT"
python -m pytest tests/test_trainer_eval_before_train.py -k awex
exec bash "$AREAL_ROOT/examples/vlm/pacman/scripts/smoke.sh" \
    --config examples/vlm/pacman/curriculum1_awex_colocate.yaml \
    environment.max_steps=2 planner_audit.max_steps=2 total_train_steps=2 "$@"
