#!/usr/bin/env bash
set -euo pipefail

variant=${1:?Usage: bash examples/weftmoe/run.sh baseline|bf16|fp8 [config overrides...]}
shift
export WEFTMOE_TRIAL="$variant"
case "$variant" in
  baseline) export AREAL_USE_WEFTMOE=0 WEFTMOE_FP8=0 ;;
  bf16) export AREAL_USE_WEFTMOE=1 WEFTMOE_FP8=0 ;;
  fp8) export AREAL_USE_WEFTMOE=1 WEFTMOE_FP8=1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
if [[ "$variant" != baseline ]]; then
  : "${WEFTMOE_TENSOR_EXTENSION:?Build WeftMoE and set WEFTMOE_TENSOR_EXTENSION}"
  [[ -f "$WEFTMOE_TENSOR_EXTENSION" ]] || {
    echo "WEFTMOE_TENSOR_EXTENSION does not name a file" >&2; exit 2;
  }
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
overrides=()
if [[ "$variant" == baseline ]]; then
  overrides+=(actor.megatron.moe_token_dispatcher_type=flex actor.megatron.moe_enable_deepep=true)
fi
exec python examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py \
  --config examples/weftmoe/dapo_math_grpo.yaml "${overrides[@]}" "$@"
