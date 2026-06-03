# Quantization

FSDP BF16 Training + SGLang FP8 Rollout

## Overview

These configs demonstrate online FP8 block-wise quantization for SGLang inference rollout while keeping FSDP training in BF16. The training engine quantizes BF16 weights to FP8 (128×128 blocks, e4m3fn) before NCCL broadcast to SGLang.

## Hardware Requirements

- CUDA GPU with FP8 compute capability (SM89+, e.g. Ada Lovelace / Hopper)
- SGLang built with FP8 support (`--quantization=fp8`)

## Configs

| Config | Engine | Task | Quantization |
|--------|--------|------|-------------|
| `fsdp_math_grpo_fp8.yaml` | FSDP | GSM8K math (GRPO) | FP8 block-wise |

### Key Parameters

#### actor

The training engine configuration.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `backend` | `fsdp:d4p1t1` | FSDP with 4 data-parallel shards, 1 pipeline stage, 1 tensor-parallel shard |
| `path` | `Qwen/Qwen2.5-1.5B-Instruct` | Model checkpoint path |
| `dtype` | `bfloat16` | Training compute dtype (BF16) |
| `weight_update_mode` | `xccl` | Use legacy xccl NCCL path for weight broadcast |

#### sglang

The inference engine configuration.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_path` | `${actor.path}` | Same model as training |
| `dtype` | `${actor.dtype}` | Same dtype as training |
| `quantization` | `fp8` | **Enable FP8 block-wise quantization in SGLang** |
| `mem_fraction_static` | `0.8` | GPU memory fraction for static allocation |
| `context_length` | `32768` | Max context length |

#### rollout

The rollout controller configuration.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `backend` | `sglang:d4p1t1` | SGLang inference with same parallel layout |
| `max_concurrent_rollouts` | `256` | Max concurrent inference requests |

## How It Works

1. **Training**: FSDPEngine keeps weights in BF16, computes gradients in BF16
2. **Weight sync**: Before each weight update broadcast, FSDPEngine all-gathers sharded weights, then quantizes eligible 2D Linear layers to FP8 with per-128×128-block scales
3. **Broadcast**: `fp8_weight` (float8_e4m3fn) and `weight_scale_inv` (float32) are broadcast separately via NCCL
4. **Rollout**: SGLang receives FP8 weights and scales, uses them directly for block-wise FP8 GEMM

## Parameters Filtered

- **Quantized**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`
- **Skipped**: `embed_tokens`, `lm_head`, `layernorm`, `norm`, `ln_`, `embeddings`, `mlp.gate.weight`

## Usage

```bash
python -m areal.examples.math.gsm8k_rl \
    --config examples/quantization/fsdp_math_grpo_fp8.yaml
```

## See Also

- `areal/utils/kernel/fp8_kernel.py` — Unified FP8 quantization kernel
