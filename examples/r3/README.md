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
