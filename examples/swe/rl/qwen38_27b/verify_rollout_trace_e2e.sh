#!/usr/bin/env bash
# Submit the existing Qwen/Arena recipe with an opt-in trace YAML selected in site.env.
# Requires the normal site profile, shared storage, Slurm, Arena, and rollout images.
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile=${QWEN_ENV_FILE:-$recipe_dir/site.env}
if [[ ! -f $profile ]]; then
  echo "Missing QWEN_ENV_FILE: $profile" >&2
  exit 2
fi

# site.env is the source of the config selection used by submit.sh.
set -a
source "$profile"
set +a
config_name=${QWEN_TRAIN_CONFIG:-train_trace.yaml}
config_path="$recipe_dir/$config_name"
if [[ $config_name != "$(basename -- "$config_name")" || ! -f $config_path ]]; then
  echo "Place an opt-in YAML in $recipe_dir and set QWEN_TRAIN_CONFIG in site.env" >&2
  exit 2
fi
if ! python3 - "$config_path" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
if not re.search(r"(?m)^  _version: v1\s*$", text):
    raise SystemExit("selected rollout config must use v1")
if not re.search(
    r"(?m)^    rollout_trace:\s*\n(?:^      .*\n)*?^      enabled: true\s*$",
    text,
):
    raise SystemExit("selected rollout config must enable rollout_trace")
PY
then
  echo "The selected YAML must enable rollout.agent.rollout_trace on v1" >&2
  exit 2
fi

expect_progress=${QWEN_EXPECT_PROGRESS_LOG:-0}
if [[ $expect_progress == 1 ]] && ! grep -Eq '^[[:space:]]+log_mode:[[:space:]]+progress([[:space:]]|$)' "$config_path"; then
  echo "Set rollout.agent.rollout_trace.log_mode: progress in $config_path" >&2
  exit 2
fi

echo "Submitting Arena end-to-end rollout trace validation with $config_name"
bash "$recipe_dir/submit.sh"

log_file="$QWEN_OUTPUT_ROOT/logs/${QWEN_RUN_USER:-$(id -un)}/$QWEN_EXPERIMENT_NAME/$QWEN_ARENA_TRIAL/main.log"
deadline=$((SECONDS + 5 * 60 * 60))
while (( SECONDS < deadline )); do
  if [[ -f $log_file ]] \
    && grep -Fq 'timeperf/rollout/schedule_requests' "$log_file" \
    && grep -Fq 'timeperf/rollout/pre_process' "$log_file" \
    && grep -Fq 'timeperf/rollout/generate' "$log_file" \
    && grep -Fq 'timeperf/rollout/post_process' "$log_file" \
    && grep -Fq 'timeperf/rollout/reward' "$log_file" \
    && grep -Fq 'timeperf/rollout/pause_wait' "$log_file" \
    && grep -Fq 'timeperf/rollout/data_collect' "$log_file" \
    && grep -Fq 'timeperf/rollout/other' "$log_file" \
    && { [[ $expect_progress != 1 ]] || { grep -Fq 'Rollout progress: running=' "$log_file" && grep -Fq 'uuid=' "$log_file"; }; }; then
    echo "Rollout timing metrics observed in $log_file"
    if [[ $expect_progress == 1 ]]; then
      echo "Check the last Rollout progress entry for group and session counts."
    else
      echo "Check the last Rollout long tails entry for Arena task_id / attempt_id."
    fi
    exit 0
  fi
  sleep 60
done
echo "Timed out waiting for first-step rollout timing metrics in $log_file" >&2
exit 1
