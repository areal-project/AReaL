# LoRA Reference

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank
matrices into pre-trained weights, typically around linear layers. Compared with
full-parameter fine-tuning, this reduces memory usage and compute cost substantially,
making RL fine-tuning of large models much more practical on limited hardware.

In AReaL, this is especially useful for:

- reinforcement learning with very large models, including 70B+ models, on relatively
  modest hardware such as 8 x 80 GB GPUs,
- enabling larger batch sizes because LoRA reduces training memory pressure,
- simplifying transfer and deployment because only the LoRA adapters need to be saved
  and shipped,
- \[Future\] fine-tune multiple LoRA adapters more efficiently in parallel for better
  hardware utilization (see RFC
  [#609](https://github.com/areal-project/AReaL/issues/609)).

This guide explains how to enable LoRA in RL training and configure the related
parameters.

## Backend Support

The current LoRA support matrix in AReaL is:

| Engine   | vLLM | SGLang |
| -------- | ---- | ------ |
| FSDP2    | ✅   | ✅     |
| Megatron | ✅   | ❌     |
| Archon   | ❌   | ❌     |

**Example scripts:**

| Engine   | Example script                                |
| -------- | --------------------------------------------- |
| FSDP2    | `examples/math/gsm8k_grpo_lora.yaml`          |
| Megatron | `examples/math/gsm8k_grpo_megatron_lora.yaml` |

### FSDP2 LoRA Example

The following example uses `examples/math/gsm8k_grpo_lora.yaml` as the reusable base
configuration. It trains Qwen3.6-27B with FSDP2 on eight NPUs and uses the other eight
NPUs for four PP2 vLLM rollout replicas. FSDP2 LoRA uses disk-based adapter updates.

**Qwen3.6-27B, single node:**

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

For Megatron + vLLM, AReaL now supports:

- LoRA fine-tuning on MoE architectures such as Qwen3 MoE with XCCL-based LoRA weight.
- Cross-node LoRA training when the Megatron and rollout groups span multiple nodes.

### Megatron LoRA Modes

The Megatron engine supports LoRA in two modes:

1. **LoRA on both Megatron and vLLM.** Megatron trains the LoRA adapters, and vLLM
   applies the adapters during rollout. This mode currently supports dense models only.
1. **LoRA on Megatron with merged weights on vLLM.** Megatron trains the LoRA adapters,
   merges them into the base weights before each weight update, and sends the merged
   weights to vLLM. vLLM therefore does not need to apply LoRA adapters during rollout.
   This mode supports both dense and MoE models.

#### Separate LoRA Examples

These examples train LoRA adapters in Megatron and apply the separate adapters in vLLM
during rollout. They use Megatron-Bridge's native adapter export for live weight
updates.

**Qwen3-0.6B, single node:**

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

**Qwen3.6-27B, single node:**

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

**Qwen3.6-27B, multiple nodes:**

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

#### Merged LoRA Examples

Use `actor.use_merged_lora=true` for this mode. The following commands use
`examples/math/gsm8k_grpo_megatron_merged.yaml` as the shared base configuration. Values
such as model paths, dataset paths, and parallelism layouts are supplied as command-line
overrides so that the base configuration remains reusable.

**Qwen3-0.6B, single node:**

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

**Qwen3-30B, single node:**

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

**Qwen3-30B, multiple nodes:**

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

**Qwen3.6-27B, single node:**

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

**Qwen3.6-27B, multiple nodes:**

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

## Core LoRA Parameters

| Parameter         | What it controls                                                                                        | Typical values        |
| ----------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| `use_lora`        | Enables LoRA fine-tuning mode.                                                                          | `true` / `false`      |
| `lora_rank` (`r`) | Rank of the low-rank adapters. Higher rank increases capacity and memory/compute cost.                  | `8`, `16`, `32`, `64` |
| `lora_alpha`      | LoRA scaling factor. Effective adapter scale is commonly thought of as proportional to `alpha / r`.     | `16`, `32`, `64`      |
| `target_modules`  | Which model submodules receive LoRA adapters. This is the most important architecture-specific setting. | e.g. \[`all-linear`\] |
| `peft_type`       | PEFT method type. In AReaL configs, this is LoRA.                                                       | `lora`                |

## Practical Notes

- Start with `r=16` or `r=32` for most models, then tune upward only if needed.
- Keep `target_modules` consistent with your model architecture naming.
- For Megatron backend, LoRA requires `megatron-bridge` instead of `mbridge`.
