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
