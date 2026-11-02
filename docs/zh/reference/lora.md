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

| Engine   | Example script                                |
| -------- | --------------------------------------------- |
| FSDP2    | `examples/math/gsm8k_grpo_lora.yaml`          |
| Megatron | `examples/math/gsm8k_grpo_megatron_lora.yaml` |

### FSDP2 LoRA 示例

以下示例使用 `examples/math/gsm8k_grpo_lora.yaml` 作为可复用的基础配置。该配置使用 8 个 NPU 通过 FSDP2 训练
Qwen3.6-27B，并使用另外 8 个 NPU 运行 4 个 PP2 vLLM rollout 副本。FSDP2 LoRA 通过磁盘更新 adapter。

**Qwen3.6-27B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_lora.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  rollout.backend=vllm:d4p2t1 \
  rollout.max_concurrent_rollouts=10000 \
  rollout.max_head_offpolicyness=0 \
  actor.backend=fsdp:d8p1t1 \
  actor.path=/data/efs/models/Qwen3.6-27B \
  +actor.fsdp.memory_efficient_load=true \
  actor.mb_spec.max_tokens_per_mb=3000 \
  ref.mb_spec.max_tokens_per_mb=3000 \
  actor.optimizer.lr=1.70e-5 \
  vllm.max_model_len=4000 \
  vllm.gpu_memory_utilization=0.97 \
  +vllm.enforce_eager=true \
  train_dataset.batch_size=256 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=256 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

对于 Megatron + vLLM，AReaL 现在支持：

- 在 Qwen3 MoE 等 MoE 架构上进行 LoRA 微调，并通过 XCCL 更新 LoRA 权重。
- 当 Megatron 与 rollout group 横跨多个节点时进行跨节点 LoRA 训练。

### Megatron LoRA 模式

Megatron engine 支持两种 LoRA 模式：

1. **Megatron 和 vLLM 均使用 LoRA。** Megatron 训练 LoRA adapter，vLLM 在 rollout 期间应用这些
   adapter。该模式目前仅支持 dense 模型。
1. **Megatron 使用 LoRA，vLLM 使用合并后的权重。** Megatron 训练 LoRA adapter，
   在每次权重更新前将其合并到基础权重中，然后将合并后的权重发送给 vLLM。因此， vLLM 在 rollout 期间不需要应用 LoRA adapter。该模式同时支持
   dense 和 MoE 模型。

#### 独立 LoRA 示例

以下示例在 Megatron 中训练 LoRA adapter，并在 rollout 期间由 vLLM 应用独立 adapter。实时权重更新使用
Megatron-Bridge 的原生 adapter 导出功能。

**Qwen3-0.6B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_lora.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  rollout.backend=vllm:d8p1t1 \
  actor.backend=megatron:d8p1t1 \
  actor.path=/data/efs/models/Qwen3-0.6B \
  actor.gradient_checkpointing=false \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  '+actor.megatron={bridge_type: megatron-bridge, use_bridge_for_update_weights: true, enable_mtp_training: false}' \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3.6-27B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_lora.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  rollout.backend=vllm:d2p4t1 \
  actor.backend=megatron:d2p4t1 \
  actor.path=/data/efs/models/Qwen3.6-27B \
  actor.gradient_checkpointing=false \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  '+actor.megatron={bridge_type: megatron-bridge, use_bridge_for_update_weights: true, enable_mtp_training: false}' \
  actor.use_lora=true \
  rollout.use_lora=true \
  "+actor.target_modules=['language_model.*.linear_qkv','language_model.*.linear_proj']" \
  vllm.enable_lora=true \
  vllm.max_lora_rank=16 \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3.6-27B，多节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_lora.yaml \
  scheduler.type=ray \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  rollout.backend=vllm:d4p4t1 \
  actor.backend=megatron:d4p4t1 \
  actor.path=/data/efs/models/Qwen3.6-27B \
  actor.gradient_checkpointing=false \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  '+actor.megatron={bridge_type: megatron-bridge, use_bridge_for_update_weights: true, enable_mtp_training: false}' \
  actor.use_lora=true \
  rollout.use_lora=true \
  "+actor.target_modules=['language_model.*.linear_qkv','language_model.*.linear_proj']" \
  vllm.enable_lora=true \
  vllm.max_lora_rank=16 \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

#### 合并 LoRA 示例

该模式需设置 `actor.use_merged_lora=true`。以下命令使用
`examples/math/gsm8k_grpo_megatron_merged.yaml` 作为共享基础配置。模型路径、
数据集路径和并行布局等参数均通过命令行覆盖，从而保持基础配置的通用性。

**Qwen3-0.6B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_merged.yaml \
  +actor.target_modules='[linear_qkv,linear_proj,linear_fc1,linear_fc2]' \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  scheduler.type=local \
  cluster.n_gpus_per_node=16 \
  rollout.backend=vllm:d8p1t1 \
  actor.backend=megatron:d8p1t1 \
  actor.path=/data/efs/models/Qwen3-0.6B \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3-30B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_merged.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=16 \
  rollout.backend=vllm:d1p1t4 \
  rollout.max_concurrent_rollouts=128 \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  "actor.backend='megatron:(attn:d1p12t1c1|ffn:d1p12t1e1)'" \
  actor.path=/home/model/Qwen3-30B-A3B-Base \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  +actor.target_modules='[linear_qkv,linear_proj]' \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  +vllm.enable_expert_parallel=true \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3-30B，多节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_merged.yaml \
  scheduler.type=ray \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime/name_resolve \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  rollout.backend=vllm:d4p1t4 \
  rollout.max_concurrent_rollouts=128 \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  "actor.backend='megatron:(attn:d1p4t4c1|ffn:d1p4t1e4)'" \
  actor.path=/home/model/Qwen3-30B-A3B-Base \
  +actor.target_modules='[linear_qkv,linear_proj]' \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  +vllm.enable_expert_parallel=true \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3.6-27B，单节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_merged.yaml \
  scheduler.type=local \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=16 \
  rollout.backend=vllm:d2p1t2 \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  actor.backend=megatron:d2p4t1 \
  actor.path=/data/efs/models/Qwen3.6-27B \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  "+actor.target_modules=['language_model.*.linear_qkv','language_model.*.linear_proj']" \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

**Qwen3.6-27B，多节点：**

```bash
python examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_megatron_merged.yaml \
  scheduler.type=ray \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=16 \
  cluster.fileroot=/data/efs/areal_runtime/name_resolve \
  cluster.name_resolve.nfs_record_root=/data/efs/areal_runtime/name_resolve \
  rollout.backend=vllm:d8p1t2 \
  +rollout.agent.admin_api_key=my-unique-admin-key-124 \
  actor.backend=megatron:d4p4t1 \
  actor.path=/data/efs/models/Qwen3.6-27B \
  actor.mb_spec.max_tokens_per_mb=5120 \
  ref.mb_spec.max_tokens_per_mb=5120 \
  "+actor.target_modules=['language_model.*.linear_qkv','language_model.*.linear_proj']" \
  vllm.max_model_len=8192 \
  vllm.gpu_memory_utilization=0.8 \
  train_dataset.batch_size=128 \
  train_dataset.path=/data/efs/datasets/gsm8k \
  valid_dataset.batch_size=128 \
  valid_dataset.path=/data/efs/datasets/gsm8k \
  2>&1 | tee out.log
```

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
