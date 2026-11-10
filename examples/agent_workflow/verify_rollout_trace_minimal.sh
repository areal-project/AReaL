#!/usr/bin/env bash
# One local GPU, a small model, two GSM8K samples, and one training step.
# Run only on a machine with the CUDA/SGLang environment already installed.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

export AREAL_TRACE_ADMIN_KEY=${AREAL_TRACE_ADMIN_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}
model_path=${AREAL_TRACE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
trial_name=${AREAL_TRACE_TRIAL:-rollout-trace-smoke-$(date +%Y%m%d%H%M%S)}
trace_root=${AREAL_TRACE_ROOT:-$(mktemp -d)}

uv run python examples/agent_workflow/train.py \
  --config examples/agent_workflow/config.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 cluster.n_gpus_per_node=1 \
  "cluster.fileroot=$trace_root" \
  "cluster.name_resolve.nfs_record_root=$trace_root/name_resolve" \
  rollout.backend=sglang:d1p1t1 actor.backend=fsdp:d1p1t1 ref.backend=fsdp:d1p1t1 \
  "actor.path=$model_path" "trial_name=$trial_name" \
  total_train_steps=1 gconfig.n_samples=1 \
  gconfig.max_new_tokens=128 gconfig.max_tokens=256 \
  train_dataset.batch_size=2 valid_dataset.batch_size=2 \
  rollout.max_concurrent_rollouts=2 rollout.consumer_batch_size=2 \
  rollout.agent.rollout_trace.enabled=true \
  rollout.agent.rollout_trace.collect_interval_seconds=1 \
  rollout.agent.rollout_trace.log_interval_seconds=1 \
  rollout.agent.rollout_trace.log_mode=progress \
  perf_tracer.session_tracer.enabled=true \
  sglang.log_requests=true sglang.log_level=info sglang.log_level_http=info \
  'rollout.agent.admin_api_key=${oc.env:AREAL_TRACE_ADMIN_KEY}'

log_file="$trace_root/logs/$(id -un)/gsm8k-grpo/$trial_name/main.log"
for metric in schedule_requests pre_process generate post_process reward pause_wait data_collect other; do
  if ! grep -Fq "timeperf/rollout/$metric" "$log_file"; then
    echo "Missing timeperf/rollout/$metric in $log_file" >&2
    exit 1
  fi
done
if ! grep -Fq 'Rollout progress: running=' "$log_file"; then
  echo "Missing live rollout progress in $log_file" >&2
  exit 1
fi
if ! grep -Fq 'uuid=' "$log_file"; then
  echo "Missing session UUID in live rollout progress" >&2
  exit 1
fi
echo "One-step rollout timing and live progress verified in $log_file"
python3 examples/agent_workflow/check_sglang_session_rids.py \
  "$trace_root/logs/$(id -un)/gsm8k-grpo/$trial_name"
