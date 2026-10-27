#!/usr/bin/env bash
# Run on a CUDA GPU host. Not executed on the development machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL_PATH:?Set MODEL_PATH to a local BF16 Qwen3.5-0.8B checkpoint}"
: "${FP8_MODEL_PATH:?Set FP8_MODEL_PATH to a new rollout checkpoint directory}"
: "${FILEROOT:?Set FILEROOT to the experiment output directory}"
export TRIAL_NAME="${TRIAL_NAME:-fp8-smoke-$(date +%Y%m%d-%H%M%S)}"
if [[ ! -f "$FP8_MODEL_PATH/config.json" ]]; then
  python -m areal.tools.prepare_qwen35_fp8 "$MODEL_PATH" "$FP8_MODEL_PATH"
fi
python examples/math/gsm8k_rl.py \
  --config examples/math/qwen3_5_fp8_grpo.yaml "$@"
