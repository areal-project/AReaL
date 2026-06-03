# 量化

FSDP BF16 训练 + SGLang FP8 推理

## 概述

这些配置展示了在保持 FSDP 训练使用 BF16 的同时，为 SGLang 推理 rollout 使用在线 FP8 分块量化。训练引擎在 NCCL 广播到 SGLang 之前，将 BF16 权重量化为 FP8（128×128 分块，e4m3fn）。

## 硬件要求

- 支持 FP8 计算能力的 CUDA GPU（SM89+，例如 Ada Lovelace / Hopper）
- 支持 FP8 的 SGLang（`--quantization=fp8`）

## 配置

| 配置 | 引擎 | 任务 | 量化 |
|------|------|------|------|
| `fsdp_math_grpo_fp8.yaml` | FSDP | GSM8K 数学 (GRPO) | FP8 分块 |

### 关键参数

#### actor

训练引擎配置。

| 参数 | 值 | 说明 |
|------|-----|------|
| `backend` | `fsdp:d4p1t1` | FSDP，4 个数据并行分片，1 个流水线阶段，1 个张量并行分片 |
| `path` | `Qwen/Qwen2.5-1.5B-Instruct` | 模型检查点路径 |
| `dtype` | `bfloat16` | 训练计算数据类型（BF16） |
| `weight_update_mode` | `xccl` | 使用 legacy xccl NCCL 路径进行权重广播 |

#### sglang

推理引擎配置。

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_path` | `${actor.path}` | 与训练使用相同模型 |
| `dtype` | `${actor.dtype}` | 与训练使用相同数据类型 |
| `quantization` | `fp8` | **在 SGLang 中启用 FP8 分块量化** |
| `mem_fraction_static` | `0.8` | 静态分配的 GPU 显存比例 |
| `context_length` | `32768` | 最大上下文长度 |

#### rollout

Rollout 控制器配置。

| 参数 | 值 | 说明 |
|------|-----|------|
| `backend` | `sglang:d4p1t1` | SGLang 推理，使用相同的并行布局 |
| `max_concurrent_rollouts` | `256` | 最大并发推理请求数 |

## 工作原理

1. **训练**：FSDPEngine 保持权重为 BF16，梯度计算使用 BF16
2. **权重同步**：在每次权重更新广播之前，FSDPEngine all-gather 分片权重，然后将符合条件的 2D Linear 层量化为 FP8，使用每 128×128 分块的缩放因子
3. **广播**：`fp8_weight`（float8_e4m3fn）和 `weight_scale_inv`（float32）通过 NCCL 分别广播
4. **推理**：SGLang 接收 FP8 权重和缩放因子，直接用于分块 FP8 GEMM

## 使用方法

```bash
python -m areal.examples.math.gsm8k_rl \
    --config examples/quantization/fsdp_math_grpo_fp8.yaml
```

## 参见

- `areal/utils/kernel/fp8_kernel.py` — 统一 FP8 量化内核
