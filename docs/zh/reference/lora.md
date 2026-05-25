# LoRA 参考

LoRA 是一种参数高效的微调技术，会在预训练权重中注入可训练的低秩矩阵， 通常作用在线性层附近。与全参数微调相比，LoRA 可以显著降低显存占用和计算开销， 从而让大模型的
RL 微调在硬件资源有限的条件下也更具可行性。

在 AReaL 中，LoRA 尤其适用于以下场景：

- 在相对有限的硬件条件下进行超大模型的强化学习训练，例如使用 8 x 80 GB GPU 训练 70B+ 规模模型，
- 由于显存压力更低，可以支持更大的 batch size，
- 模型迁移与部署更加简单，因为只需要保存和分发 LoRA adapter，
- \[Future\] 更高效地并行微调多个 LoRA adapter，以提升硬件利用率（参见 RFC
  [#609](https://github.com/areal-project/AReaL/issues/609)）。

本文档说明如何在 RL 训练中启用 LoRA，并配置相关参数。

## 后端支持

AReaL 当前的 LoRA 支持矩阵如下：

| Engine   | vLLM | SGLang |
| -------- | ---- | ------ |
| FSDP2    | ✅   | ✅     |
| Megatron | ✅   | ❌     |
| Archon   | ❌   | ❌     |

**示例脚本：**

| Engine       | Example script                                    |
| ------------ | ------------------------------------------------- |
| FSDP2        | `examples/math/gsm8k_grpo_lora.yaml`              |
| Megatron     | `examples/math/gsm8k_grpo_megatron_lora.yaml`     |
| Megatron MoE | `examples/math/gsm8k_grpo_megatron_lora_moe.yaml` |

对于 Megatron + vLLM，AReaL 现在支持：

- 在 Qwen3 MoE 等 MoE 架构上进行 LoRA 微调，并通过 XCCL 更新 LoRA 权重。
- 当 Megatron 与 rollout group 横跨多个节点时进行跨节点 LoRA 训练。

## 核心 LoRA 参数

| 参数              | 作用                                                               | 常见取值              |
| ----------------- | ------------------------------------------------------------------ | --------------------- |
| `use_lora`        | 是否启用 LoRA 微调模式。                                           | `true` / `false`      |
| `lora_rank` (`r`) | 低秩适配器的秩。`r` 越大，表达能力越强，但显存与计算开销更高。     | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA 缩放系数。通常可理解为有效缩放与 `alpha / r` 成正比。         | `16`, `32`, `64`      |
| `target_modules`  | 指定注入 LoRA 的目标子模块。这是最关键、且与模型结构强相关的配置。 | 例如 \[`all-linear`\] |
| `peft_type`       | PEFT 方法类型。在 AReaL 配置中为 LoRA。                            | `lora`                |

## 实践建议

- 可先从 `r=16` 或 `r=32` 开始，再按效果和资源逐步调参。
- `target_modules` 需与具体模型的层命名保持一致。
- 对于 Megatron 后端，LoRA 需要使用 `megatron-bridge`，而不是 `mbridge`。
