## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

The hyperparameters given in gsm8k_grpo.yaml is the set that we found to achieve the
highest max `grpo-eval/task_reward/avg` during training for `Qwen2.5-1.5b-Instruct`. You
are free to try out more of the hyperparameters listed below!

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

### Other Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)

## Awex Example Location

Awex-specific GSM8K sample scripts were moved to:

- `examples/experimental/awex/README.md`
- `examples/experimental/awex/gsm8k_grpo_awex_sample.yaml`

Awex meta server bootstrap is now handled by `PPOTrainer` when
`actor.weight_update_mode=awex`; the standard `examples/math/gsm8k_rl.py` entrypoint can
be used directly with the AWEX sample yaml. Auto-start is only available in
single-controller mode; SPMD runs must provide an explicit `awex.meta_server_addr`.

## Score centering with vLLM

Score centering subtracts an estimate of the sampler-expected score from the policy
gradient, so training survives the drift between the vLLM sampler and the FSDP learner.
This branch carries the sampler's top-k token ids and log-probabilities through the
**OpenAI agent transport** (`ArealOpenAI` proxy), which is how `gsm8k_rl.py` drives
rollouts here; the native SGLang metadata path is not used on this backend.

Start from an existing recipe with these overrides, and set model paths, resource
allocation and output directories for your environment:

```bash
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo_npu.yaml scheduler.type=local \
  actor.use_score_centering=true actor.recompute_logprob=false \
  actor.use_decoupled_loss=false actor.rejection_sampling=null \
  gconfig.score_centering_top_k=128 gconfig.top_k=-1 gconfig.top_p=1.0 \
  gconfig.temperature=1.0
```

Enabling score centering selects token-wise REINFORCE with a detached head/tail
correction, replacing the clipped PPO surrogate. Score centering owns its head
weighting, so `actor.rejection_sampling` and `actor.importance_sampling` must stay
`null`; `sc_mode` selects the weight function applied to the sampled-token ratio `p/q`
and to the tail mass ratio:

```yaml
actor:
  use_score_centering: true
  recompute_logprob: false
  use_decoupled_loss: false
  rejection_sampling: null
  importance_sampling: null
  sc_mode: tis            # none (default) | tis | mis
  sc_clip_low: 0.0        # lower band bound; 0 disables the lower clamp
  sc_clip_high: 2.0       # upper bound, must be finite for tis/mis
```

- `none` leaves the head weight at one: pure centering.
- `tis` clips the ratio into `[sc_clip_low, sc_clip_high]`.
- `mis` keeps the ratio inside that band and zeroes it outside.

`tis` requires `0 <= sc_clip_low < sc_clip_high` with a finite upper bound; `mis`
additionally requires `sc_clip_high > 0`. The correction carries the matching head/tail
weights, so the sampled term is not weighted twice. Diagnostics are exported as
`sc_correction`, `sc_loss_term`, `sc_tis_weight`, `sc_tis_clamped_fraction`, `sc_rho`,
`sc_q_mass` and `sc_p_mass`.

Support requires the vLLM rollout backend, temperature 1, untruncated sampling and no
penalties. On this backend score centering additionally requires full-vocab,
full-sequence trainer logits, so tensor parallel with a vocabulary-sharded LM head and
Ulysses/context parallel fail fast. SAPO/CISPO, decoupled PPO, tree training and
distillation remain unsupported, and `score_centering_top_k` never reaches the OpenAI
SDK (the proxy consumes it).
