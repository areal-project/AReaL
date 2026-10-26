# R3 Math Examples

This directory contains local Moonlight examples for R3 router replay with Megatron
training and SGLang rollout.

- `gsm8k_rlvr.py`: single-turn GSM8K RLVR with `areal.workflow.rlvr.RLVRWorkflow`.
- `gsm8k_multiturn.py`: GSM8K multi-turn correction loop through `ArealOpenAI`, exported
  as interaction trajectories.

Both configs use one node with 8 GPUs split into 4 rollout GPUs and 4 actor GPUs:

```bash
rollout.backend=sglang:d1p1t4
actor.backend='megatron:(attn:d1p1t4|ffn:d1p1t1e4)'
```

The default config enables R3:

```bash
export MOONLIGHT_MODEL_PATH=/workspace/models/Moonlight-16B-A3B-Instruct
export GSM8K_DATA_PATH=/storage/openpsi/data/gsm8k

python -m examples.r3.gsm8k_rlvr \
  --config examples/r3/gsm8k_rlvr_moonlight.yaml

python -m examples.r3.gsm8k_multiturn \
  --config examples/r3/gsm8k_multiturn_moonlight.yaml
```

To run the same examples without R3, disable routed-expert collection and Megatron
router replay:

```bash
python -m examples.r3.gsm8k_rlvr \
  --config examples/r3/gsm8k_rlvr_moonlight.yaml \
  trial_name=no-r3 \
  rollout.return_routed_experts=false \
  actor.megatron.enable_router_replay=false

python -m examples.r3.gsm8k_multiturn \
  --config examples/r3/gsm8k_multiturn_moonlight.yaml \
  trial_name=no-r3 \
  rollout.return_routed_experts=false \
  actor.megatron.enable_router_replay=false
```

The examples assume:

- `MOONLIGHT_MODEL_PATH` points to a local Moonlight checkpoint, for example
  `/workspace/models/Moonlight-16B-A3B-Instruct`.
- `GSM8K_DATA_PATH` points to a local GSM8K dataset, for example
  `/storage/openpsi/data/gsm8k`.
- SGLang supports returning routed experts for the local MoE model.

The committed configs are short smoke runs (`total_train_steps=3`,
`train_dataset.batch_size=8`, `gconfig.n_samples=4`). Override those fields for longer
comparisons.

## Routing memory

The built-in workflows export routing as a packed CPU tensor
`routed_experts[sum(sequence_lengths), moe_layers, topk]`, together with
`r3_cu_seqlens[batch_size + 1]` and `r3_routing_valid[batch_size]`. Expert IDs use
signed int16 when lossless, including models with 512 experts, and otherwise int32.
Legacy padded routing inputs remain accepted.

PPO minibatch permutations and engine reordering operate on row metadata instead of
copying the routing payload. RPC transfers stream only each physical pipeline stage's
layers, including the union of its virtual pipeline chunks. CP/SP token selection and
the conversion to native int64 replay indices happen on CPU for the current microbatch
before upload. Routing is excluded from advantage computation, reference, critic, and
teacher payloads.

The engine logs host RSS/PSS before replay and exports `r3/cpu_routing_storage_bytes`,
counting unique replay backing storages rather than counting each microbatch view again.
This metric excludes model weights and other process allocations; use host PSS and
container memory measurements for a complete node budget.

For an isolated actor-side memory comparison, run these in separate processes:

```bash
python tests/torchrun/run_r3_memory_probe.py --format dense
python tests/torchrun/run_r3_memory_probe.py --format packed
```

The packed case starts after PP4 distribution. The probe measures routing preparation,
including retained input trajectories, rather than model training or the source worker's
transfer peak.
