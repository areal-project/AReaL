#!/usr/bin/env bash
# One-GPU VLM smoke run with two GRPO steps. Not executed locally.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FP8_CONFIG="${SCRIPT_DIR}/qwen3_5_2b_megatron_geometry3k_grpo_smoke.yaml"
export TRIAL_NAME="${TRIAL_NAME:-fp8-vlm-smoke-$(date +%Y%m%d-%H%M%S)}"
exec bash "${SCRIPT_DIR}/run_qwen35_fp8.sh" "$@"
