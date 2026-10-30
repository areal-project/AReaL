## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

`gsm8k_grpo.yaml` uses `6e-6`, retuned for the current FSDP FP32-master optimizer path.

### Current FP32-Master Recipe

A seed-1 run on 8 NVIDIA A800 GPUs completed the official 10-epoch schedule with the
following held-out evaluation results:

| lr     | weight decay | group size | best eval reward | final eval reward |
| ------ | ------------ | ---------- | ---------------- | ----------------- |
| 6.0E-6 | 0.017        | 4          | **0.78412**      | 0.77767           |

### Historical Pre-FP32-Master Sweep

The results below were collected with BF16 parameter and optimizer-state storage. They
are retained for reference and are not directly comparable with the current recipe.

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

#### Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)

## Score centering with SGLang

`gsm8k_score_centering.py` uses the native single-turn `RLVRWorkflow` to carry SGLang's
sampler head IDs and probabilities into FSDP training. Unlike `gsm8k_rl.py`, it does not
use the OpenAI agent transport, which does not carry these heads.

Start from the existing recipe with these overrides, and set model paths, resource
allocation and output directories for your environment:

```bash
python3 examples/math/gsm8k_score_centering.py \
  --config examples/math/gsm8k_grpo.yaml scheduler.type=local \
  actor.use_score_centering=true actor.recompute_logprob=false \
  actor.use_decoupled_loss=false actor.rejection_sampling=null \
  gconfig.score_centering_top_k=128 gconfig.top_k=-1 gconfig.top_p=1.0 \
  gconfig.temperature=1.0 actor.temperature=1.0
```

Enabling score centering selects unweighted REINFORCE with a detached head/tail
correction, replacing the clipped PPO surrogate. It does not apply TIS or rejection
weights. The reference calculation uses ordinary PyTorch log-softmax and gather with
checkpointed chunks, without a custom backward or kernel. Full input logits are still
retained. No new dependency or SGLang runtime patch is required.

Initial support is ordinary FSDP sequences with TP=SP=1, native SGLang rollout,
temperature 1, and no sampling truncation or penalties. SAPO/CISPO, decoupled PPO,
rejection sampling, tree training, OpenAI agent workflows and distillation are
unsupported. Evaluation disables head collection. Other training paths retain their
existing behavior when score centering is disabled.

Tests cover dense-reference gradients, one Adam update, interrupted rollout metadata and
padded minibatches. A single-node Qwen3-0.6B smoke exercises real SGLang heads through
the actor and FSDP optimizer. A paper-style Countdown fixed-weight-noise experiment
compares plain policy gradient with reference SC; see
[reproduction notes](score_centering_countdown.md). Multi-node training has not been
validated. This feature makes no performance claim.
