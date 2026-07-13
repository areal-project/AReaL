# Fine-tuning Large MoE Models

Compared to PyTorch FSDP, Megatron-LM supports full 5D parallelism, delivering better
scaling and efficiency. AReaL fully supports customized RL training with Megatron-LM as
the backend. This guide explains how to harness the Megatron training backend and train
large MoE models for your application.

## Enabling Megatron Backend

Shifting from FSDP to Megatron requires only a single line of change: the
`actor.backend` field from `fsdp:d4` to `megatron:d4`.

For a complete guide on allocation mode syntax, parallelism dimensions, and GPU
calculations, see the [Allocation Mode Reference](../reference/alloc_mode.md).

## Bridge Backend Selection

`MegatronEngine` supports two bridge backends configured by
`actor.megatron.bridge_type`:

```yaml
actor:
    megatron:
        bridge_type: mbridge  # default (backward compatible)
```

Set `bridge_type: megatron-bridge` to use the new backend.

For trade-offs and migration guidance, see the
[Megatron Bridge Backend Reference](../reference/bridge_backend.md).

## MoE Parallel Strategy

For MoE models, Megatron supports separate parallelism for attention and FFN modules
using the hybrid syntax. For example:

```
megatron:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

This 16-GPU configuration uses PP=4, with attention modules using TP=2 and CP=2, while
expert modules use TP=1 and EP=4. See
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)
for details on this feature.

**Tuning Guides:**

- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
- [verl with Megatron Practice](https://github.com/ISEEKYAN/verl_megatron_practice)

## Aligning Inference and Training Precision

Due to the sparse nature of MoE models, the logits calculated by forward passes during
inference and training could be severely misaligned, leading to unstable training
results. To mitigate this instability, it is highly recommended to set
`actor.megatron.use_deterministic_algorithms=True` to disable nondeterministic
calculations in Megatron, although this may cause a ~10-20% slowdown in training steps.

As an example, you can run GRPO on the Qwen3 30B-A3B MoE model and GSM8K dataset (on a
32-GPU ray cluster) directly with the following command:

```bash
# NOTE: Allocation mode here is only for illustration purposes. It is not optimized.
python3 examples/math/gsm8k_rl.py --config <megatron_config.yaml> \
    scheduler.type=ray \
    experiment_name=megatron-moe-gsm8k-grpo trial_name=trial-0 \
    rollout.backend=sglang:d4t4 actor.backend=megatron:(attn:d1p4t2c2|ffn:d1p4t1e4) \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 actor.path=Qwen/Qwen3-30B-A3B \
    actor.megatron.use_deterministic_algorithms=True
```
