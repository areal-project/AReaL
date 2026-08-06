(section-optimizer-guide)=

# 优化器配置指南

AReaL 支持多种优化器类型，可通过 `optimizer.type` 字段进行配置。本文档介绍各优化器在不同训练后端的支持情况，以及 Muon 优化器的实现差异。

## 支持的优化器类型

| 类型        | 说明                                                                            |
| ----------- | ------------------------------------------------------------------------------- |
| `adam`      | AdamW 优化器（默认）                                                            |
| `adam_bf16` | BF16 精度的 AdamW，降低优化器状态显存占用                                       |
| `sgd`       | 标准 SGD                                                                        |
| `muon`      | Muon 优化器，对 ≥2D 参数使用 Newton-Schulz 正交化更新，\<2D 参数使用 AdamW 后端 |

## 各引擎支持矩阵

| 优化器      |      FSDP Engine       |        Megatron Engine         | Archon Engine |
| ----------- | :--------------------: | :----------------------------: | :-----------: |
| `adam`      |           ✅           |               ✅               |      ✅       |
| `adam_bf16` | ✅ (AnyPrecisionAdamW) | ✅ (precision-aware optimizer) |      ❌       |
| `sgd`       |           ✅           |               ✅               |      ✅       |
| `muon`      |           ✅           |   ✅ (Megatron-Core ≥ 0.17)    |  ❌ (未实现)  |

### 备注

- **FSDP Engine** 中 `adam_bf16` 使用 `AnyPrecisionAdamW`，将 momentum 和 variance 存储为 BF16。
- **Megatron Engine** 中 `adam_bf16` 要求模型 dtype 为 bfloat16，会自动转换为 adam 并启用
  precision-aware optimizer。
- **Archon Engine** 目前仅支持 `adam` 和 `sgd`，Muon 支持尚在开发中。

## Muon 优化器

### 简介

Muon (MomentUm Orthogonalized by Newton-schulz) 是一种利用 Newton-Schulz
迭代对梯度动量进行近似正交化的优化器。其核心思想是：对权重矩阵的梯度施加正交约束，使更新方向在参数空间中更加"均匀"，从而加速收敛。

### 参考实现与论文

| 资源                                     | 链接                                               |
| ---------------------------------------- | -------------------------------------------------- |
| 原始实现 (Keller Jordan)                 | https://github.com/KellerJordan/Muon               |
| Moonlight 论文 (RMS scaling)             | https://arxiv.org/abs/2502.16982                   |
| AReaL FSDP 实现                          | `areal/engine/fsdp_utils/muon.py`                  |
| Emerging-Optimizers (Megatron-Core Muon) | https://github.com/NVIDIA-NeMo/Emerging-Optimizers |

### FSDP 与 Megatron 实现差异

FSDP Engine 和 Megatron Engine 对 Muon 的参数分组策略存在显著差异：

| 维度               | FSDP Engine                                              | Megatron Engine                                                          |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Muon 参数范围**  | **所有** ≥2D 参数（包括 embedding 权重矩阵）             | **Linear 层的 weight**                                                   |
| **AdamW 后端参数** | 所有 \<2D 参数（bias、LayerNorm weight/bias）            | embedding、bias、norm 以及非 Linear 的 2D 参数                           |
| **分布式 NS 实现** | DTensor gather/scatter（FSDP2 原生）                     | TP-aware 的 `TensorParallelMuon`（利用 TP 通信组做分布式 Newton-Schulz） |
| **TP + EP 支持**   | TP + FSDP 2D mesh ✅；TP + EP + FSDP 3D mesh ❌ (未实现) | 完整支持 TP / EP / PP                                                    |

### 配置示例

```yaml
optimizer:
  type: muon
  lr: 2e-3                    # 统一 lr（Muon 和 AdamW 后端共用）
  muon_momentum: 0.95
  muon_use_nesterov: true
  muon_num_ns_steps: 5
  muon_scale_mode: spectral      # spectral / unit_rms_norm / shape_scaling
  muon_extra_scale_factor: 0.2   # 0.2 + spectral 等价于 Moonlight 风格的 RMS 对齐
  weight_decay: 0.05
  beta1: 0.9                  # AdamW 后端参数
  beta2: 0.95
  eps: 1e-5
  lr_scheduler_type: cosine
  warmup_steps_proportion: 0.03
```

### 配置参数说明

| 参数                      | 类型  | 默认值             | 说明                                                                                                                                                                                                    |
| ------------------------- | ----- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`                      | float | 0.001              | 统一学习率，Muon（≥2D 参数）和 AdamW 后端（\<2D 参数）共用。配合 `muon_scale_mode=spectral` + `muon_extra_scale_factor=0.2`（Moonlight 风格）时单一 lr 即可                                             |
| `muon_momentum`           | float | 0.95               | Muon 动量系数                                                                                                                                                                                           |
| `muon_use_nesterov`       | bool  | true               | 是否使用 Nesterov 动量                                                                                                                                                                                  |
| `muon_num_ns_steps`       | int   | 5                  | Newton-Schulz 迭代步数                                                                                                                                                                                  |
| `muon_scale_mode`         | str   | "spectral"         | 更新缩放模式。`spectral`：`sqrt(max(m, n))`（Kimi/Moonlight、emerging_optimizers 默认）；`unit_rms_norm`：`sqrt(m / n)`（Scion / Bernstein）；`shape_scaling`：`max(1, m/n)**0.5`（Keller Jordan 原版） |
| `muon_extra_scale_factor` | float | 1.0                | 额外乘性缩放，最终 scale = `scale_factor(mode) * muon_extra_scale_factor`。配合 `spectral` 使用 `0.2` 可复刻 Moonlight 风格的 RMS 对齐缩放                                                              |
| `weight_decay`            | float | 0.01               | 权重衰减，同时作用于 Muon 和 AdamW 后端                                                                                                                                                                 |
| `beta1` / `beta2` / `eps` | float | 0.9 / 0.999 / 1e-8 | AdamW 后端的超参数                                                                                                                                                                                      |
