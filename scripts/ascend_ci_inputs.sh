# Input preparation shared by the Ascend PR and nightly training jobs.
resolve_cached_qwen_model() {
    local model_name=$1 override=${2:-}
    local encoded_name=${model_name//./___}
    local ms_root="${HOME}/.cache/modelscope/hub/models"
    local hf_root="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
    local candidate weight
    local -a candidates=(
        "$override"
        "$ms_root/Qwen/$model_name" "$ms_root/Qwen/$encoded_name"
        "$ms_root/$model_name" "$ms_root/$encoded_name"
        "$hf_root/models--Qwen--$model_name/snapshots/"*
    )
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" && -f "$candidate/config.json" ]] || continue
        weight=$(find -L "$candidate" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit)
        if [[ -n "$weight" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "::error title=Cached model unavailable::$model_name requires config.json and a top-level .safetensors or .bin weight file" >&2
    printf 'Checked: %s\n' "${candidates[@]}" >&2
    return 1
}

check_geometry3k_splits() {
    python - "$@" <<'PY'
import sys
from datasets import load_dataset

for split in sys.argv[2:]:
    dataset = load_dataset(sys.argv[1], split=split)
    if not len(dataset):
        raise ValueError(f"Empty Geometry3K split: {split}")
    print(f"Cached Geometry3K {split}: {len(dataset)} examples")
PY
}

prepare_geometry3k() {
    local attempt
    if HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 check_geometry3k_splits "$@"; then
        return 0
    fi
    for attempt in 1 2 3; do
        if HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 check_geometry3k_splits "$@"; then
            return 0
        fi
        if [[ "$attempt" -lt 3 ]]; then sleep "$((attempt * 30))"; fi
    done
    echo "::error::Geometry3K preparation failed after 3 attempts" >&2
    return 1
}
