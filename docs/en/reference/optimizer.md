(section-optimizer-guide)=

# Optimizer Configuration Guide

AReaL supports multiple optimizer types, configurable via the `optimizer.type` field.
This document covers the support matrix across training backends and the implementation
differences of the Muon optimizer.

## Supported Optimizer Types

| Type        | Description                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------- |
| `adam`      | AdamW optimizer (default)                                                                          |
| `adam_bf16` | BF16-precision AdamW, reduces optimizer state memory                                               |
| `sgd`       | Standard SGD                                                                                       |
| `muon`      | Muon optimizer: Newton-Schulz orthogonalized updates for ≥2D params, AdamW backend for \<2D params |

## Engine Support Matrix

| Optimizer   |      FSDP Engine       |        Megatron Engine         |    Archon Engine     |
| ----------- | :--------------------: | :----------------------------: | :------------------: |
| `adam`      |           ✅           |               ✅               |          ✅          |
| `adam_bf16` | ✅ (AnyPrecisionAdamW) | ✅ (precision-aware optimizer) |          ❌          |
| `sgd`       |           ✅           |               ✅               |          ✅          |
| `muon`      |           ✅           |   ✅ (Megatron-Core ≥ 0.17)    | ❌ (not implemented) |

### Notes

- **FSDP Engine**: `adam_bf16` uses `AnyPrecisionAdamW`, storing momentum and variance
  in BF16.
- **Megatron Engine**: `adam_bf16` requires model dtype to be bfloat16; it is
  auto-converted to adam with precision-aware optimizer enabled.
- **Archon Engine**: Currently only supports `adam` and `sgd`. Muon support is under
  development.

## Muon Optimizer

### Overview

Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer that applies approximate
orthogonalization to gradient momentum via Newton-Schulz iteration. The core idea is to
impose an orthogonal constraint on weight matrix gradients, making update directions
more "uniform" in parameter space and accelerating convergence.

### Reference Implementations and Papers

| Resource                                 | Link                                               |
| ---------------------------------------- | -------------------------------------------------- |
| Original implementation (Keller Jordan)  | https://github.com/KellerJordan/Muon               |
| Moonlight paper (RMS scaling)            | https://arxiv.org/abs/2502.16982                   |
| AReaL FSDP implementation                | `areal/engine/fsdp_utils/muon.py`                  |
| Emerging-Optimizers (Megatron-Core Muon) | https://github.com/NVIDIA-NeMo/Emerging-Optimizers |

### FSDP vs Megatron Implementation Differences

The FSDP Engine and Megatron Engine differ significantly in how they partition
parameters for Muon:

| Dimension                         | FSDP Engine                                                       | Megatron Engine                                                                       |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Muon parameter scope**          | **All** ≥2D parameters (including embedding weight matrices)      | **Linear layer weights**                                                              |
| **AdamW backend parameters**      | All \<2D parameters (bias, LayerNorm weight/bias)                 | Embeddings, biases, norms, and non-Linear 2D parameters                               |
| **Distributed NS implementation** | DTensor gather/scatter (FSDP2 native)                             | TP-aware `TensorParallelMuon` (distributed Newton-Schulz over TP communication group) |
| **TP + EP support**               | TP + FSDP 2D mesh ✅; TP + EP + FSDP 3D mesh ❌ (not implemented) | Full TP / EP / PP support                                                             |

### Configuration Example

```yaml
optimizer:
  type: muon
  lr: 2e-3                    # Shared lr (Muon and AdamW backend)
  muon_momentum: 0.95
  muon_use_nesterov: true
  muon_num_ns_steps: 5
  muon_scale_mode: spectral      # spectral / unit_rms_norm / shape_scaling
  muon_extra_scale_factor: 0.2   # 0.2 + spectral = Moonlight-style RMS-matched scaling
  weight_decay: 0.05
  beta1: 0.9                  # AdamW backend params
  beta2: 0.95
  eps: 1e-5
  lr_scheduler_type: cosine
  warmup_steps_proportion: 0.03
```

### Configuration Parameters

| Parameter                 | Type  | Default            | Description                                                                                                                                                                                                        |
| ------------------------- | ----- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lr`                      | float | 0.001              | Shared learning rate for both Muon (≥2D params) and AdamW backend (\<2D params). A single lr works well when pairing `muon_scale_mode=spectral` with `muon_extra_scale_factor=0.2` (Moonlight-style)               |
| `muon_momentum`           | float | 0.95               | Muon momentum coefficient                                                                                                                                                                                          |
| `muon_use_nesterov`       | bool  | true               | Whether to use Nesterov momentum                                                                                                                                                                                   |
| `muon_num_ns_steps`       | int   | 5                  | Number of Newton-Schulz iteration steps                                                                                                                                                                            |
| `muon_scale_mode`         | str   | "spectral"         | Update scaling mode. `spectral`: `sqrt(max(m, n))` (Kimi/Moonlight, emerging_optimizers default). `unit_rms_norm`: `sqrt(m / n)` (Scion / Bernstein). `shape_scaling`: `max(1, m/n)**0.5` (Keller Jordan original) |
| `muon_extra_scale_factor` | float | 1.0                | Extra multiplicative scale; final scale = `scale_factor(mode) * muon_extra_scale_factor`. Use `0.2` with `spectral` to reproduce Moonlight-style RMS-matched scaling                                               |
| `weight_decay`            | float | 0.01               | Weight decay, applied to both Muon and AdamW backend                                                                                                                                                               |
| `beta1` / `beta2` / `eps` | float | 0.9 / 0.999 / 1e-8 | AdamW backend hyperparameters                                                                                                                                                                                      |
