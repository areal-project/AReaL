#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
# Paths and resource placement are supplied by submit_rl.sh.
exec srun --mpi=none --ntasks=1 --cpus-per-task=4 --mem=10G \
  singularity exec --no-eval --pid --writable-tmpfs \
  --bind "$QWEN_CONTROLLER_MOUNTS" \
  --env "PYTHONPATH=$QWEN_CONTROLLER_PYTHONPATH" \
  "$QWEN_ACTOR_IMAGE" bash \
  "$QWEN_REPO/examples/swe/qwen38_flash_next/rl_controller.sh" "$@"
