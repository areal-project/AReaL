#!/usr/bin/env bash
set -euo pipefail
suite=$1
work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/ascend-nightly-XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
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
        model=${GRPO_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}
        ;;
    sft)
        config=tests/configs/ascend_nightly_geometry3k_sft.yaml
        entry=examples.vlm.geometry3k_sft
        model=${SFT_MODEL_PATH:-Qwen/Qwen3.6-27B}
        ;;
    *) exit 2 ;;
esac
# AWEX and the Megatron bridge both need a local HF model directory.
if [[ ! -d "$model" ]]; then
    model=$(python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$model")
fi
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
python -m "$entry" "${args[@]}"
python -m scripts.ascend_nightly_verify "$work" "$experiment" "$suite" nightly-output/metrics.json
