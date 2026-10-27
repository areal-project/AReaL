#!/usr/bin/env bash
# Qwen3.5-2B Geometry3K VLM training on a CUDA host. Not executed locally.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL_PATH:?Set MODEL_PATH to a complete local BF16 Qwen3.5-2B VLM checkpoint}"
: "${FP8_MODEL_PATH:?Set FP8_MODEL_PATH to the separate FP8 rollout checkpoint directory}"
: "${FILEROOT:?Set FILEROOT to the experiment output directory}"
export TRIAL_NAME="${TRIAL_NAME:-fp8-vlm-$(date +%Y%m%d-%H%M%S)}"
FP8_CONFIG="${FP8_CONFIG:-examples/fp8/qwen3_5_2b_megatron_geometry3k_grpo.yaml}"
if [[ ! -f "$FP8_MODEL_PATH/config.json" ]]; then
  python -m examples.fp8.prepare_qwen35_fp8 "$MODEL_PATH" "$FP8_MODEL_PATH"
fi
python examples/vlm/geometry3k_grpo.py --config "$FP8_CONFIG" "$@"
