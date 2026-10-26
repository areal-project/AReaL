#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

: "${FILEROOT:?Set FILEROOT to a writable experiment root}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rl-perf-trace}
TRIAL_NAME=${TRIAL_NAME:-smoke-$(date +%Y%m%d-%H%M%S)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
ACTOR_BACKEND=${ACTOR_BACKEND:-fsdp:d1p1t1}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang:d1p1t1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
N_SAMPLES=${N_SAMPLES:-1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}
PROFILE_STEPS=${PROFILE_STEPS:-'[1]'}

AREAL_TORCH_PROFILER_PROFILE_MEMORY=false \
python -m areal.infra.launcher.local examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  experiment_name="$EXPERIMENT_NAME" \
  trial_name="$TRIAL_NAME" \
  cluster.fileroot="$FILEROOT" \
  cluster.name_resolve.nfs_record_root="$FILEROOT/name_resolve/$TRIAL_NAME" \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node="$N_GPUS_PER_NODE" \
  actor.backend="$ACTOR_BACKEND" \
  rollout.backend="$ROLLOUT_BACKEND" \
  actor.path="$MODEL_PATH" \
  train_dataset.batch_size="$TRAIN_BATCH_SIZE" \
  valid_dataset.batch_size="$TRAIN_BATCH_SIZE" \
  train_dataset.max_length=128 \
  gconfig.n_samples="$N_SAMPLES" \
  gconfig.max_new_tokens=32 \
  gconfig.max_tokens=256 \
  actor.mb_spec.max_tokens_per_mb=512 \
  rollout.max_concurrent_rollouts=4 \
  sglang.context_length=1024 \
  total_train_steps="$TOTAL_TRAIN_STEPS" \
  saver.freq_epochs=null \
  saver.freq_steps=null \
  recover.freq_epochs=null \
  recover.freq_steps=null \
  evaluator.freq_epochs=null \
  evaluator.freq_steps=null \
  stats_logger.wandb.mode=disabled \
  perf_tracer.enabled=true \
  perf_tracer.save_interval=10 \
  perf_tracer.profile_steps="$PROFILE_STEPS" \
  perf_tracer.session_tracer.enabled=true

echo "Trace directory: $FILEROOT/logs/$(whoami)/$EXPERIMENT_NAME/$TRIAL_NAME"
