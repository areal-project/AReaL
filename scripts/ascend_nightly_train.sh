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
python -m "$entry" "${args[@]}"
python -m scripts.ascend_nightly_verify "$work" "$experiment" "$suite" nightly-output/metrics.json
