# Quantization Examples

FSDP BF16 Training + SGLang FP8 Rollout

## Overview

These configs demonstrate online FP8 block-wise quantization for SGLang inference rollout while keeping FSDP training in BF16. The training engine quantizes BF16 weights to FP8 (128x128 blocks, e4m3fn) before NCCL broadcast to SGLang.

## Configs

| Config | Engine | Task | Quantization |
|--------|--------|------|-------------|
| `fsdp_math_grpo_fp8.yaml` | FSDP | GSM8K math (GRPO) | FP8 block-wise |

## How It Works

1. **Training**: FSDPEngine keeps weights in BF16, computes gradients in BF16
2. **Weight sync**: Before each weight update broadcast, FSDPEngine all-gathers sharded weights, then quantizes eligible 2D Linear layers to FP8 with per-128x128-block scales
3. **Broadcast**: `fp8_weight` (float8_e4m3fn) and `weight_scale_inv` (float32) are broadcast separately via NCCL
4. **Rollout**: SGLang receives FP8 weights and scales, uses them directly for block-wise FP8 GEMM

## Parameters Filtered

Quantized: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`

Skipped: `embed_tokens`, `lm_head`, `layernorm`, `norm`, `ln_`, `embeddings`, `mlp.gate.weight` (MoE router)

## Usage

```bash
python -m areal.examples.math.gsm8k_rl \
    --config examples/quantization/fsdp_math_grpo_fp8.yaml
```

## Requirements

- SGLang with FP8 support (`--quantization=fp8`)
- CUDA GPU with FP8 compute capability (SM89+)
- `sglang.quantization: fp8` in config

## See Also

- `areal/utils/kernel/fp8_kernel.py` - Unified FP8 quantization kernel
- `docs/superpowers/plans/2026-05-30-fsdp-sglang-fp8-rollout-proposal.md` - Design proposal
