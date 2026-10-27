#!/usr/bin/env bash
# Eight-GPU VLM run: training TP=2, inference TP=4, four AWEX update steps.
# Not executed locally.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FP8_CONFIG="${SCRIPT_DIR}/qwen3_5_2b_megatron_geometry3k_grpo.yaml"
export TRIAL_NAME="${TRIAL_NAME:-fp8-vlm-e2e-$(date +%Y%m%d-%H%M%S)}"
exec bash "${SCRIPT_DIR}/run_qwen35_fp8.sh" \
  total_train_steps=4 \
  train_dataset.batch_size=8 \
  rollout.max_concurrent_rollouts=8 \
  gconfig.n_samples=2 \
  gconfig.max_new_tokens=256 \
  sglang.context_length=4096 \
  sglang.max_running_requests=8 \
  evaluator.freq_epochs=null \
  saver.freq_epochs=null \
  "$@"
