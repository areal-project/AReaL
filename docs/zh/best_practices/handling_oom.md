# 处理 OOM 问题

OOM 错误在大规模 RL 训练中很常见。本指南介绍如何在 AReaL 的生成、训练和权重更新阶段解决这些问题。

## 理解内存使用情况

在应用修复之前，了解哪些参数影响内存使用：

### 核心参数

- **各引擎的 `backend` 字段（如 `actor.backend`、`rollout.backend`）**：推理和训练如何在 GPU
  之间分配。对于大模型，张量并行通常比数据并行每个 GPU 使用更少的内存。

- **`train_dataset.max_length`**：最大提示长度。更长的提示需要更多内存。

- **`gconfig.max_new_tokens`**：每个提示生成的 token 数。与 `max_length` 结合使用，这决定了总序列长度。

- **`actor.mb_spec.max_tokens_per_mb`**：前向/后向传递期间每个微批的 token 数。这是控制训练内存的主要参数。不能设置为低于
  `max_length + max_new_tokens`。

- **`max_concurrent_rollouts`**：并行生成请求的数量。更多请求可以提高吞吐量，但会增加内存使用。

### 引擎特定参数

- **推理引擎**：`sglang.mem_fraction_static` 控制 SGLang 使用多少 GPU 内存。查看
  [SGLang 文档](https://docs.sglang.io/) 了解更多调优选项。

- **训练引擎**：FSDP 分片和其他 PyTorch
  设置也会影响内存使用。[FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html) 有更多详细信息。

> 注意：`train_dataset.batch_size` 不影响峰值内存使用。排查 OOM 问题时，请专注于上述参数。

## 解决生成 OOM 错误

当发生生成 OOM 错误时，请尝试以下解决方案：

### 1. 减少并发 Rollouts（最有效）

降低并行生成请求的数量：

```yaml
max_concurrent_rollouts: 200  # Try reducing from default values like 256
```

这直接减少了推理服务器的内存压力，通常是最有效的解决方案。

### 2. 调整并行策略

增加张量并行以将模型权重分配到更多 GPU：

```yaml
# Before: 4 data parallel processes for rollout
# After: 2 data parallel, 2 tensor parallel for rollout
rollout:
  backend: "sglang:d2t2"
actor:
  backend: "fsdp:d4"
```

请注意，较高的张量并行会降低生成吞吐量。

### 3. 调整 SGLang 参数

调整 SGLang 内存分配：

```yaml
sglang:
  mem_fraction_static: 0.8  # Reduce from 0.9 to leave more memory headroom
```

查看 [SGLang 文档](https://docs.sglang.io/) 了解更多调优选项。

## 解决训练 OOM 错误

训练 OOM 错误需要减少梯度计算和模型更新的内存占用。

### 1. 优化微批大小

将 `max_tokens_per_mb` 设置到尽可能低：

```yaml
actor:
  mb_spec:
    max_tokens_per_mb: 4096  # train_dataset.max_length + gconfig.max_new_tokens
```

对于多轮对话，计算方式如下：

```
max_tokens_per_mb = <longest_conversation_length> + gconfig.max_new_tokens
```

确切值取决于您的 `RolloutWorkflow` 实现。

### 2. 启用梯度检查点

```yaml
actor:
  gradient_checkpointing: true
```

### 3. 启用 5D 并行

对于无法进一步降低 `max_tokens_per_mb` 的长上下文场景，使用 Ulysses 序列并行将序列分配到多个 GPU：

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel, 2 ulysses context parallel for training
rollout:
  backend: "sglang:d4"
actor:
  backend: "fsdp:d2c2"
```

> Ulysses 上下文并行大小必须能整除模型的注意力头数量。
>
> 例如，对于 40 个注意力头：
>
> - 有效：`1, 2, 4, 8`
> - 无效：`16, 32`

您也可以使用 FSDP 启用张量并行：

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel, 2 tensor parallel for training
rollout:
  backend: "sglang:d4"
actor:
  backend: "fsdp:d2t2"
```

对于 Megatron 和 Archon 后端，您还可以启用流水线和专家并行：

```yaml
# Before: 4 data parallel processes for training
# After: 2 data parallel with 2 overlaid expert parallel, 2 pipeline parallel, still 4 GPUs
rollout:
  backend: "sglang:d4"
actor:
  backend: "archon:d2p2e2"
```

我们推荐使用流水线和专家并行而不是张量/上下文并行。查看[分配模式参考文档](../reference/alloc_mode.md)了解更多详情。

### 4. 启用逐层优化器步进

当使用 FSDP CPU 卸载（`offload_params: true`）时，默认的 CPU Adam 步进可能非常慢。
启用逐层优化器步进，将优化器状态逐层流式传输到设备以获得显著加速：

```yaml
actor:
  fsdp:
    per_layer_optim_step: true
    optim_step_prefetch_layers: 1  # 预取层数（默认：1）
```

这将逐层流式传输状态到设备执行 Adam 更新，而非在 CPU 上运行， 保持设备内存占用低的同时实现更快的优化器更新。

**前置条件：**

- 同时兼容 `offload_params: true` 和 `false`（优化器状态由逐层包装器自动管理在 CPU 上； 当 `offload_params`
  也启用时，参数/梯度也会逐层流式传输到设备）

### 5. 切换到轻量级优化器

AReaL 根据训练引擎支持不同的优化器。

| 优化器        | FSDP | Megatron | 名称      |
| ------------- | ---- | -------- | --------- |
| AdamW（默认） | ✅   | ✅       | adam      |
| SGD           | ✅   | ✅       | sgd       |
| AdamW_bf16    | ✅   | ❌       | adam_bf16 |

`SGD` 和 `AdamW_bf16` 比默认的 `AdamW` 使用更少的内存。通过在 YAML 配置文件中设置
`actor.optimizer.type: <name>` 来切换（例如 `actor.optimizer.type: sgd`）。

### 6. 使用内存高效模型加载

如果在模型初始化期间（训练开始前）发生 OOM，请启用内存高效加载：

```yaml
actor:
  fsdp:
    memory_efficient_load: true
```

这对于直接将完整权重加载到每个 GPU 会超出内存的大模型很有用。启用后：

1. 所有 ranks 在 CPU 上创建模型结构（不加载 LLM 的权重）
1. 应用 FSDP 并行化
1. Rank 0 加载预训练权重并广播到所有 ranks
1. 权重转移到 GPU

这种方法以一些初始化时间为代价，显著降低了模型加载期间的峰值 GPU 内存。

**视觉-语言模型（VLMs）的注意事项**：VLMs 不使用 rank 0 广播优化。当为 VLM 设置 `memory_efficient_load: true`
时，权重在 CPU 上加载而不是 GPU，但每个 rank 独立加载权重。这仍然可以减少初始化期间的 GPU 内存使用，但不会减少 CPU 内存或磁盘/网络 I/O。

## 解决权重更新 OOM 错误

权重更新消耗大量内存，尤其是在使用 NCCL 同步（默认设置）时。

### 1. 切换到基于磁盘的更新

从 NCCL 切换到基于磁盘的权重同步：

```yaml
actor:
  weight_update_mode: disk
```

确保 `cluster.fileroot` 是集群中的共享目录。

### 2. 减少内存缓冲区大小

要继续使用 NCCL，请减少权重分块的内存缓冲区大小：

```python
# In WeightUpdateMeta.from_fsdp_xccl() calls
WeightUpdateMeta.from_fsdp_xccl(
    ...,
    weight_chunked_mem_mb = 512,  # Reduce from default (typically 1024+)
)
```

## 减少优化器状态显存占用

FSDP 后端默认维护 fp32 主权重以及 fp32 的 AdamW 优化器状态 （`exp_avg`、`exp_avg_sq`），与 DeepSpeed ZeRO-3 和
Megatron 的 precision-aware optimizer 行为一致。对于 `N` 十亿参数的模型， 所有 GPU 总计约占用 `12N` GB：

| 组件                        | Bytes/param |
| --------------------------- | ----------- |
| fp32 主权重（storage）      | 4           |
| fp32 `exp_avg`（一阶矩）    | 4           |
| fp32 `exp_avg_sq`（二阶矩） | 4           |

**CPU 内存提示**：开启 `actor.fsdp.memory_efficient_load=true` 时， rank 0 会先在 CPU
上加载完整模型再广播。fp32 storage 让这一峰值翻倍 — 8B 模型的 rank 0 CPU 占用会从 ~16 GB 上升到 ~32 GB。请相应规划主节点
内存，或设置 `memory_efficient_load=false` 让所有 rank 分摊 CPU 负载。

**DCP checkpoint 提示**：训练 checkpoint（Distributed Checkpoint） 保留 storage
dtype（fp32），以保证恢复时主权重精度正确。HF 导出和 rollout 权重同步**始终**会 cast 回 compute dtype，所以部署侧仍为 bf16。

如果遇到 OOM 又无法增加并行度，可以切换为 bf16 优化器状态 + Kahan summation 更新：

```yaml
actor:
  dtype: bfloat16
  optimizer_dtype: bfloat16   # storage 使用 bf16，与 AdamW state 对齐
  optimizer:
    type: adam_bf16           # 使用 AnyPrecisionAdamW + Kahan summation
```

这可节省约 `8N` GB：

| 组件                           | Bytes/param |
| ------------------------------ | ----------- |
| bf16 主权重                    | 2           |
| bf16 `exp_avg`                 | 2           |
| bf16 `exp_avg_sq`              | 2           |
| bf16 Kahan compensation buffer | 2           |

依据 AnyPrecision 论文，Kahan summation 能恢复 fp32 等效的更新精度。 单 step 时间相比 fused fp32 AdamW 慢约
5-10%；在 dense 和 MoE 模型上 收敛质量相当。

**不要将 `optimizer.type: adam` 与 `optimizer_dtype: bfloat16` 同时使用** — `torch.optim.AdamW`
会静默创建 bf16 优化器状态，后期 loss 会比 fp32 主权重的方案高约 3 倍（参见 issue #1292）。运行时检测到该组合会输出 warning。
