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

使用 `qwen3_5` 架构的 Qwen GDN 混合模型，需要 Megatron-Core >=0.18.2 和 Megatron-Bridge >=0.5.1 才能使用
packed THD 与 CP。 在这些版本上，AReaL 将 MTP 标签和回答 mask 与每条 packed 序列的 CP 分片对齐。 旧版运行时保留 padded
路径并要求 `CP=1`。仓库默认依赖版本不会自动升级；仅更换配置不能让旧版本支持 THD/CP。 Dense Qwen 的 EP 为 1。

针对部分 Megatron-Core 版本的 MTP 完整重计算兼容补丁支持缺省 padding mask； 如果上游 checkpoint 实现无法传递非空 padding
mask，则显式报错，不会静默丢弃 mask。
