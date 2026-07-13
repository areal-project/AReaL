# 大型 MoE 模型微调

与 PyTorch FSDP 相比，Megatron-LM 支持完整的 5D 并行性，提供更好的扩展性和效率。AReaL 完全支持使用 Megatron-LM
作为后端进行自定义 RL 训练。本指南将解释如何利用 Megatron 训练后端，并为您的应用训练大型 MoE 模型。

## 启用 Megatron 后端

从 FSDP 切换到 Megatron 只需要更改一行：将 `actor.backend` 字段从 `fsdp:d4` 改为 `megatron:d4`。

有关分配模式语法、并行维度 和 GPU 计算的完整指南，请参阅[分配模式参考](../reference/alloc_mode.md)。

## Bridge 后端选择

`MegatronEngine` 支持通过 `actor.megatron.bridge_type` 选择两种 bridge 后端：

```yaml
actor:
    megatron:
        bridge_type: mbridge  # 默认（向后兼容）
```

设置 `bridge_type: megatron-bridge` 可启用新后端。

关于取舍与迁移建议，请参阅 [Megatron Bridge 后端参考](../reference/bridge_backend.md)。

## MoE 并行策略

对于 MoE 模型，Megatron 使用混合语法支持注意力模块和 FFN 模块的独立并行。例如：

```
megatron:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

这个 16-GPU 配置使用 PP=4，注意力模块使用 TP=2 和 CP=2，而专家模块使用 TP=1 和
EP=4。请参阅[MoE 并行折叠](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)了解此功能的详细信息。

**调优指南：**

- [Megatron 性能最佳实践](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
- [verl 与 Megatron 实践](https://github.com/ISEEKYAN/verl_megatron_practice)

## 对齐推理和训练精度

由于 MoE 模型的稀疏性，推理和训练期间前向传递计算的 logits 可能会严重不对齐，导致训练不稳定。为了缓解这种不稳定，强烈建议设置
`actor.megatron.use_deterministic_algorithms=True` 以禁用 Megatron 中的非确定性计算，尽管这可能会导致训练步骤减慢约
10-20%。

例如，您可以直接使用以下命令在 Qwen3 30B-A3B MoE 模型和 GSM8K 数据集上运行 GRPO（在 32-GPU ray 集群上）：

```bash
# 注意：此处的分配模式仅用于说明目的，未经过优化。
python3 examples/math/gsm8k_rl.py --config <megatron_config.yaml> \
    scheduler.type=ray \
    experiment_name=megatron-moe-gsm8k-grpo trial_name=trial-0 \
    rollout.backend=sglang:d4t4 actor.backend=megatron:(attn:d1p4t2c2|ffn:d1p4t1e4) \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 actor.path=Qwen/Qwen3-30B-A3B \
    actor.megatron.use_deterministic_algorithms=True
```
