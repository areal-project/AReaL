#!/usr/bin/env bash
# Runs inside the training SIF. It never invokes Docker or Singularity.
set -euo pipefail
umask 077
cd "${TASK_CODE_ROOT:?}"
TASK_PYTHON=$(command -v "${TASK_PYTHON:-python3}")
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AREAL_HOME="$TASK_RUN_ROOT/areal-home"
export HERMES_HOME="$TASK_RUN_ROOT/hermes-home"
export AREAL_TASK_TOOL_ROOT="$TASK_RUN_ROOT/agent-tools"
export AREAL_TASK_STATE_ROOT="$TASK_RUN_ROOT/train/state"
export AREAL_TASK_RUN_ROOT="$TASK_RUN_ROOT/train"
export AREAL_ROLLOUT_TRACE_DIR="$TASK_RUN_ROOT/rollout-trace"
export AREAL_TASK_PYTHON_IMAGE="${AREAL_TASK_PYTHON_IMAGE:-python:3.12}"
export AREAL_CHECKPOINT_ESTIMATED_BYTES=0
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HERMES_HOME" "$AREAL_TASK_TOOL_ROOT" "$AREAL_ROLLOUT_TRACE_DIR"
printf 'model:\n  context_length: 65536\n' > "$HERMES_HOME/config.yaml"
unset HERMES_MAX_TURNS AREAL_CC_MAX_TURNS

"$TASK_PYTHON" -m examples.v2_online_workflows.environments.singularity health \
  > "$TASK_RUN_ROOT/host-health.json"
"$TASK_PYTHON" -m examples.v2_online_workflows.environments.python \
  > "$TASK_RUN_ROOT/python-tool-probe.json"
if [[ "$TASK_MODE" == probe ]]; then exit 0; fi
if [[ "$TASK_MODE" == reference ]]; then
  : "${TASK_EVAL_TASKS:?}" "${AREAL_SWE_EVALUATOR_ROOT:?}"
  "$TASK_PYTHON" -m examples.v2_online_workflows.benchmarks.swe_rebench grade-reference \
    --tasks "$TASK_EVAL_TASKS" --output "$TASK_RUN_ROOT/reference-grades" \
    > "$TASK_RUN_ROOT/reference-grades.log" 2>&1
  exit 0
fi
[[ "$TASK_MODE" == train || "$TASK_MODE" == eval ]]
export AREAL_MODEL_PATH="${TASK_MODEL_PATH:?}"
: "${AREAL_AGENT_ADMIN_KEY:?}" "${AREAL_INFERENCE_ADMIN_KEY:?}"
export AREAL_ACTOR_ADMIN_KEY="${AREAL_ACTOR_ADMIN_KEY:-$AREAL_INFERENCE_ADMIN_KEY}"
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.device_count() == 8, "Expected 8 allocated GPUs"; print(torch.cuda.get_device_name(0))' \
  > "$TASK_RUN_ROOT/gpu-probe.txt"
case "$TASK_CASE" in
  hermes-dapo|cc-dapo) benchmark=dapo_math; config=dapo.yaml ;;
  hermes-swe|cc-swe) benchmark=swe_rebench; config=swe.yaml ;;
  *) exit 2 ;;
esac
case "$TASK_CASE" in
  hermes-*) agent_class=examples.v2_online_workflows.service.TaskAgent ;;
  cc-*) agent_class=examples.v2_online_workflows.service.CCTaskAgent ;;
esac
service="sif-${SLURM_JOB_ID}-${TASK_CASE}"
trainer_pid=""
cleanup() {
  "$TASK_PYTHON" -m areal.v2.cli agent stop --service "$service" --grace-period 5 \
    >> "$TASK_RUN_ROOT/cleanup.log" 2>&1 || true
  if [[ "$TASK_MODE" == eval ]]; then
    "$TASK_PYTHON" -m areal.v2.cli inf stop --service "$service-inf" --grace 10 \
      >> "$TASK_RUN_ROOT/cleanup.log" 2>&1 || true
  fi
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
CUDA_VISIBLE_DEVICES="" "$TASK_PYTHON" -m areal.v2.cli agent run \
  --service "$service" --agent "$agent_class" --num-pairs 1 --session-timeout 7200 \
  --admin-api-key "$AREAL_AGENT_ADMIN_KEY" > "$TASK_RUN_ROOT/agent.log" 2>&1
export AREAL_AGENT_URL
AREAL_AGENT_URL=$("$TASK_PYTHON" -m areal.v2.cli agent status --service "$service" --json \
  | "$TASK_PYTHON" -c 'import json,sys; s=json.load(sys.stdin); assert s["running"],s; print(s["gateway_url"])')

export TASK_N="${TASK_N:-16}" TASK_EPOCHS="${TASK_EPOCHS:-1}" TASK_BATCH="${TASK_BATCH:-4}"
export TASK_CONCURRENCY="${TASK_CONCURRENCY:-4}"
"$TASK_PYTHON" - <<'PY'
import json, os
from pathlib import Path
n, epochs, batch = (int(os.environ[k]) for k in ("TASK_N", "TASK_EPOCHS", "TASK_BATCH"))
assert min(n, epochs, batch) > 0 and n * epochs % batch == 0
row = {k: os.environ.get(k) for k in ("TASK_CASE", "TASK_MODE", "TASK_MODEL_PATH", "TASK_TRAIN_TASKS", "TASK_EVAL_TASKS", "TASK_HW", "TASK_N", "TASK_EPOCHS", "TASK_BATCH", "TASK_CONCURRENCY")}
row["updates"] = n * epochs // batch
row["task_proc_mode"] = json.loads(Path(os.environ["TASK_RUN_ROOT"], "host-health.json").read_text())["proc_mode"]
Path(os.environ["TASK_RUN_ROOT"], "launch.json").write_text(json.dumps(row, indent=2))
PY
updates=$((TASK_N * TASK_EPOCHS / TASK_BATCH))
save_every="${TASK_SAVE_EVERY:-25}"
if ((save_every > updates)); then save_every=$updates; fi
export AREAL_INFERENCE_URL
if [[ "$TASK_MODE" == train ]]; then
  : "${TASK_TRAIN_TASKS:?}"
  CUDA_VISIBLE_DEVICES="" "$TASK_PYTHON" -m examples.v2_online_workflows.train \
    --config "examples/v2_online_workflows/configs/$config" \
    "trial_name=$service" scheduler.type=local elastic.enabled=false \
    "total_train_epochs=$TASK_EPOCHS" "total_train_steps=$updates" \
    "train_dataset.batch_size=$TASK_BATCH" actor.optimizer.lr=1e-6 \
    rollout.backend=sglang:d4 "rollout.max_concurrent_rollouts=$TASK_CONCURRENCY" \
    'rollout.agent.reasoning_parser=""' +rollout.agent.engine_max_tokens=65536 \
    +sglang.dp_size=1 sglang.mem_fraction_static=0.5 \
    'elastic.rollout_devices=[4,5,6,7]' "saver.freq_steps=$save_every" \
    perf_tracer.enabled=true +perf_tracer.save_interval=1 perf_tracer.session_tracer.enabled=true \
    > "$TASK_RUN_ROOT/trainer.log" 2>&1 &
  trainer_pid=$!
  printf '%s\n' "$trainer_pid" > "$TASK_RUN_ROOT/trainer.pid"
  ready=0
  for ((attempt=0; attempt<1800; attempt++)); do
    kill -0 "$trainer_pid" 2>/dev/null || exit 1
    if AREAL_INFERENCE_URL=$("$TASK_PYTHON" - <<'PY'
import os, re
from pathlib import Path
text = Path(os.environ["TASK_RUN_ROOT"], "trainer.log").read_text()
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
urls = re.findall(r"RolloutControllerV2[^\n]*Gateway: (http://\S+)", text)
if not urls: raise SystemExit(1)
print(urls[-1])
PY
    ); then ready=1; break; fi
    sleep 1
  done
  [[ "$ready" == 1 ]]
  tasks="$TASK_TRAIN_TASKS"
else
  : "${TASK_EVAL_TASKS:?}"
  port=$("$TASK_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  AREAL_INFERENCE_URL="http://127.0.0.1:$port"
  "$TASK_PYTHON" -m areal.v2.cli inf run --service "$service-inf" --detach \
    --scheduler local --port "$port" --admin-api-key "$AREAL_INFERENCE_ADMIN_KEY" \
    --model task-model --backend sglang:d8 --model-path "$AREAL_MODEL_PATH" \
    --tokenizer-path "$AREAL_MODEL_PATH" \
    --engine-args '--context-length 65536 --mem-fraction-static 0.5 --dtype bfloat16' \
    --proxy-args "--tool-call-parser qwen --reasoning-parser '' --engine-max-tokens 65536 --request-timeout 1800" \
    > "$TASK_RUN_ROOT/inference.log" 2>&1
  tasks="$TASK_EVAL_TASKS"
fi
client_mode=eval; model=task-model
if [[ "$TASK_MODE" == train ]]; then client_mode=resident; model="$AREAL_MODEL_PATH"; fi
for ((epoch=1; epoch<=TASK_EPOCHS; epoch++)); do
  set +e
  "$TASK_PYTHON" -m examples.v2_online_workflows.client --benchmark "$benchmark" \
    --tasks "$tasks" --output "$TASK_RUN_ROOT/client-epoch-$epoch" \
    --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
    --model "$model" --mode "$client_mode" --limit "$TASK_N" \
    --concurrency "$TASK_CONCURRENCY" --request-timeout 7200 \
    > "$TASK_RUN_ROOT/client-epoch-$epoch.log" 2>&1
  client_rc=$?
  set -e
  printf '%s\n' "$client_rc" > "$TASK_RUN_ROOT/client-epoch-$epoch.exit"
  [[ "$client_rc" == 0 ]]
done
if [[ "$TASK_MODE" == train ]]; then
  set +e
  wait "$trainer_pid"
  trainer_rc=$?
  set -e
  printf '%s\n' "$trainer_rc" > "$TASK_RUN_ROOT/trainer.exit"
  trainer_pid=""
  exit "$trainer_rc"
fi
