#!/usr/bin/env bash
set -euo pipefail
suite=$1
work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/ascend-nightly-XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
# Keep model and dataset downloads shared while giving SwanLab a private cache.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export XDG_CACHE_HOME="$work/cache"
mkdir -m 700 "$XDG_CACHE_HOME"
export AREAL_NIGHTLY_ADMIN_KEY
AREAL_NIGHTLY_ADMIN_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
echo "::add-mask::$AREAL_NIGHTLY_ADMIN_KEY"
python - <<'PY'
import torch
import torch_npu

if torch.npu.device_count() != 16:
    raise SystemExit("This benchmark requires 16 visible NPUs")
PY
case "$suite" in
    grpo)
        export USE_OPTIMIZED_MODEL=0
        config=tests/configs/ascend_nightly_geometry3k_grpo.yaml
        entry=examples.vlm.geometry3k_agent_grpo
        model_name=Qwen3.5-35B-A3B
        override=${GRPO_MODEL_PATH:-}
        valid_split=test
        ;;
    sft)
        config=tests/configs/ascend_nightly_geometry3k_sft.yaml
        entry=examples.vlm.geometry3k_sft
        model_name=Qwen3.6-27B
        override=${SFT_MODEL_PATH:-}
        valid_split=validation
        ;;
    *) exit 2 ;;
esac
source scripts/ascend_ci_inputs.sh
model=$(resolve_cached_qwen_model "$model_name" "$override")
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
prepare_geometry3k "${DATASET_PATH:-hiyouga/geometry3k}" train "$valid_split"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
experiment="ascend-nightly-geometry3k-$suite"
args=(
    --config "$config"
    "experiment_name=$experiment" trial_name=nightly
    "cluster.fileroot=$work" "cluster.name_resolve.nfs_record_root=$work/names"
    "actor.path=$model"
)
if [[ -n "${DATASET_PATH:-}" ]]; then
    args+=("train_dataset.path=$DATASET_PATH" "valid_dataset.path=$DATASET_PATH")
fi
run_user=$(python -c 'import getpass; print(getpass.getuser())')
log_dir="$work/logs/$run_user/$experiment/nightly"
mkdir -p "$log_dir"
# The controller opens main.log in append mode. Follow it from the first line.
touch "$log_dir/main.log"
python -u -m "$entry" "${args[@]}" >"$work/console.log" 2>&1 &
trainer_pid=$!
(
    previous_main_size=-1
    previous_console_size=0
    quiet_minutes=0
    while sleep 60 && kill -0 "$trainer_pid" 2>/dev/null; do
        main_size=$(stat -c %s "$log_dir/main.log")
        console_size=$(stat -c %s "$work/console.log")
        if [[ "$main_size" == "$previous_main_size" ]]; then
            quiet_minutes=$((quiet_minutes + 1))
            echo "Trainer PID $trainer_pid: main.log unchanged ($main_size bytes), console.log $console_size bytes"
            if [[ "$console_size" != "$previous_console_size" ]]; then
                tail -n 10 "$work/console.log"
            fi
            if (( quiet_minutes == 5 || quiet_minutes == 6 )); then
                echo "Controller stack after $quiet_minutes quiet minutes:"
                timeout 15s py-spy dump --nonblocking --pid "$trainer_pid" ||
                    echo "Could not sample the controller stack (status $?)."
            fi
        else
            quiet_minutes=0
        fi
        previous_main_size=$main_size
        previous_console_size=$console_size
    done
) &
monitor_pid=$!
tail --pid="$trainer_pid" -n +1 -F "$log_dir/main.log"
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
wait "$trainer_pid" || {
    status=$?
    echo "Training exited with status $status. Last trainer console messages:" >&2
    tail -n 5000 "$work/console.log" >&2
    exit "$status"
}
python -m scripts.ascend_nightly_verify "$work" "$experiment" "$suite" nightly-output/metrics.json
