# Megatron Bridge 后端

AReaL 目前为 `MegatronEngine` 支持两种 bridge 后端：

- `mbridge`（默认）
- `megatron-bridge`

可通过以下配置选择后端：

```yaml
actor:
  megatron:
    bridge_type: mbridge
```

- 使用 `bridge_type=megatron-bridge` 启用新路径。
- 未显式配置时，默认使用 `mbridge`。

## 为什么需要这个功能

- `mbridge` 正在被弃用，且不支持 PEFT/LoRA。
- `megatron-bridge` 支持更多（更新）模型架构。
- `megatron-bridge` 提供内置 PEFT/LoRA 实现。

## 建议

- 对新的 GPU 训练工作流，优先使用 `megatron-bridge`。
- 为兼容现有流程和依赖环境，短期内继续保留 `mbridge`。
- 若使用磁盘进行权重广播，建议使用 `mbridge`，其 HF 模型加载/保存实现更快、更优化。
- 若使用 XCCL 进行权重广播，加载/保存耗时影响较小。

## 当前限制

`MegatronEngine` 的 tree-attention 训练路径目前仅支持 `mbridge`，暂不支持 `megatron-bridge`。

由于 `mbridge` 在 HF 模型加载/保存上更快，在基于磁盘的工作流中仍是实用选择。

## 仅训练 MTP 层

要让原生 MTP 层适配已经微调的主模型，可在现有 SFT trainer 中配置：

```yaml
actor:
  megatron:
    bridge_type: megatron-bridge
    enable_mtp: true
    enable_mtp_training: true
    mtp_only: true
    mtp_loss_scaling_factor: 0.1
```

`mtp_only` 在分布式包装和 optimizer 创建前冻结所有非 MTP 参数，包括共享 embedding 和输出投影。 普通 SFT loss
仍保留在计算图中，用于触发 Megatron-Core 的 MTP 辅助反向传播，但不会更新冻结的主干。 不要对整个 forward 使用 `no_grad()`。loss
系数仍会缩放 MTP 梯度，必须为有限正数。

输入 checkpoint 必须同时包含目标主模型、原生 MTP 权重和匹配的模型配置。关闭 MTP 后导出的 checkpoint 需要先补回 MTP
权重及配置；补回时应保留微调后的主干、embedding、输出 head 和 tokenizer。

恢复接受率时，建议使用冻结的目标模型在代表性请求上生成的 assistant 续写，沿用 SFT 的回答 mask。 验证 MTP 参数更新且所有非 MTP
参数不变后，再在留出请求上测试接受率和吞吐；SFT loss 下降不等于推测解码一定加速。

当前要求单层 MTP 和 `megatron-bridge` 后端。MTP 之前的流水线阶段保持完全冻结，但仍参与流水线调度和优化器全局统计。 不支持
LoRA、critic、FSDP 包装及 MoE router expert-bias 更新。MTP 训练仍不支持分块 LM-head loss。
训练需要完整的主干前向计算。设置 `mtp_only: false` 则保留默认的联合训练行为。

原生 MTP 训练要求 Megatron-Core >=0.19.0 和 Megatron-Bridge >=0.6.0。在该版本组合中，Megatron-Core
从布局对齐后的 input IDs 派生 MTP targets，AReaL 则将回答 mask 与每条 packed 序列的 CP 分片对齐。 模型自主管理的 Qwen
THD forward 会在多模态 embedding 融合后，将最终 CP 切分交给 Megatron-Bridge。 Dense Qwen 的 EP 为 1。
