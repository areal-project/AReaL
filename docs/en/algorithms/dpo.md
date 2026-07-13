# Direct Preference Optimization (DPO)

Last updated: Apr 22, 2026

## Overview

**Direct Preference Optimization (DPO)** is an offline alignment algorithm that directly optimizes a language model on human preference data (chosen / rejected pairs), without a separate reward model or online RL rollouts.

Compared to RLHF (PPO), DPO is simpler (no reward model, no value network, no online generation), more stable (a single supervised-style loss), and more efficient (only two forward + one backward per batch: policy + reference). AReaL implements DPO on top of FSDP2 with reference-model colocation.

**Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., NeurIPS 2023)

## Core Idea

### The DPO Objective

Given a preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ where $y_w$ is the chosen response and $y_l$ is the rejected one, DPO optimizes:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) =
-\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[\log \sigma\!\left(\beta \left(
\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
- \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}
\right)\right)\right]
$$

$\pi_\theta$ is the policy under training, $\pi_{\text{ref}}$ the frozen reference, and $\beta$ controls the KL penalty. The objective is derived by substituting the closed-form optimal policy of KL-regularized RLHF into the Bradley-Terry preference model — the reward is implicitly defined by the policy and reference, eliminating the need for a standalone reward model.

AReaL supports two loss variants via `loss_type`: the original sigmoid form (default) and **IPO** (Azar et al. 2023), which replaces the sigmoid with a squared loss targeting a fixed margin of $\frac{1}{2\beta}$ per token. The IPO variant normalizes logratios by completion length (per-token average) before computing the squared loss, matching TRL's author-confirmed convention.

### Implicit Reward

During training we monitor the implicit reward $r(x, y) = \beta (\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x))$. A positive **reward margin** $r(x, y_w) - r(x, y_l) > 0$ indicates the model correctly prefers the chosen response; **reward accuracy** is the fraction of pairs with positive margin.

## Running the Example

### Single-Node (HH-RLHF)

```bash
python3 examples/alignment/hhrlhf_dpo.py \
  --config examples/alignment/hhrlhf_dpo.yaml \
  scheduler.type=local
```

Key fragments of `examples/alignment/hhrlhf_dpo.yaml`:

```yaml
actor:
  backend: "fsdp:d8p1t1"
  path: Qwen/Qwen2.5-7B            # Follows the original paper: train on a base model
  beta: 0.1                        # KL penalty
  dtype: bfloat16
  disable_dropout: true            # Required for DPO stability
  mb_spec:
    granularity: 2                 # Must be 2: chosen + rejected dispatched as pairs
  optimizer:
    lr: 5e-6
    lr_scheduler_type: cosine
    warmup_steps_proportion: 0.1

ref:
  backend: ${actor.backend}
  path: ${actor.path}
  optimizer: null                  # Frozen
  scheduling_strategy:
    type: colocation
    target: actor                  # Share GPUs with actor

train_dataset:
  batch_size: 8
  path: Anthropic/hh-rlhf
  type: dpo
  max_length: 2048
```

`get_hhrlhf_dpo_dataset` (`areal/dataset/hhrlhf.py`) tokenizes raw chosen/rejected text directly and infers the prompt boundary as the longest common token prefix. HH-RLHF pairs share the same multi-turn prompt and differ only in the final assistant reply, so the common prefix is exactly the prompt.

### Multi-Node (Ray)

```bash
python3 examples/alignment/hhrlhf_dpo.py \
  --config examples/alignment/hhrlhf_dpo.yaml \
  cluster.n_nodes=2 cluster.n_gpus_per_node=8 \
  cluster.fileroot=/path/to/nfs \
  scheduler.type=ray
```

## Key Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `actor.beta` | `0.1` | KL penalty. Higher values stay closer to the reference. Typical range: 0.05–0.5. |
| `actor.loss_type` | `"sigmoid"` | Loss variant. `"sigmoid"` is the original DPO; `"ipo"` uses a per-token-averaged squared loss (Azar et al. 2023). |
| `actor.optimizer.lr` | `5e-6` | Learning rate. DPO is LR-sensitive; 5e-7 – 5e-6 is the sweet spot. |
| `actor.disable_dropout` | `true` | Disable dropout for deterministic log-prob computation. |
| `actor.mb_spec.granularity` | `2` | Micro-batch granularity. Must be 2 for DPO (chosen + rejected are paired). |
| `ref` | — | Reference model configuration (required). |

Metrics `dpo/loss`, `dpo/chosen_reward`, `dpo/rejected_reward`, `dpo/reward_accuracy`, `dpo/reward_margin` are logged under the `dpo/` prefix.

## References

- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Azar et al. (2023). *A General Theoretical Paradigm to Understand Learning from Human Feedback*. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
