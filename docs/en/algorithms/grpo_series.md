# PPO, GRPO, and Related Algorithms

Last updated: Jan 4, 2026

Authors: [Ziyi ZENG](https://github.com/ZiyiTsang),
[Wei Fu](https://github.com/garrett4wade), [Honghua DONG](https://github.com/dhh1995),
[Bruce Wu](https://github.com/Bruce-rl-hw), [Bruce Li](https://github.com/HsiaoTsan)

This document covers a family of PPO-like reinforcement learning algorithms for LLM
training, including:

- **Vanilla PPO**
- **GRPO** (DeepSeekMath): [Paper](https://arxiv.org/pdf/2402.03300)
- **Dr.GRPO**: [Paper](https://arxiv.org/abs/2503.20783)
- **LitePPO**: [Paper](https://arxiv.org/pdf/2508.08221v1)
- **RLOO**: [Paper](https://arxiv.org/abs/2402.14740)
- **DAPO**: [Paper](https://arxiv.org/abs/2503.14476)
- **SAPO**: [Paper](https://arxiv.org/abs/2511.20347)
- **GSPO** (Qwen3): [Paper](https://arxiv.org/abs/2507.18071),
  [Blog](https://qwenlm.github.io/blog/gspo/)

These algorithms share the same base objective but differ in their normalization
strategies, clipping mechanisms, importance sampling levels, etc. By adjusting a few
configuration parameters in AReaL, you can switch between different algorithms.

## Example Usage

All algorithms use the same execution pattern. We recommend modifying parameters in the
configuration YAML file.

| Backend   | Command                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------- |
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=local` |
| **ray**   | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=ray`   |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=slurm` |

Replace `<algo>` with: `ppo`, `grpo`, `drgrpo`, `liteppo`, `rloo`, `gspo`,
`dapo_dynamic_bs`, or `sapo`.

### Switching Algorithms via CLI Overrides

You can also switch algorithms by overriding configuration parameters:

```bash
# Dr.GRPO from GRPO config
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  actor.adv_norm.mean_level=group \
  actor.adv_norm.std_level=null

# GSPO from GRPO config
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  +actor.importance_sampling_level=sequence

# SAPO from GRPO config
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  +actor.use_sapo_loss=true \
  +actor.sapo_tau_pos=1.0 \
  +actor.sapo_tau_neg=1.05 \
  actor.use_decoupled_loss=false
```

Note: Use `+` prefix when adding keys not present in the original YAML.

## Core Configuration Parameters

All configurations are defined in `areal/api/cli_args.py` under `PPOActorConfig` and
`NormConfig`. See [CLI configurations](../cli_reference.md) for full details.

### Reward and Advantage Normalization (`actor.reward_norm` and `actor.adv_norm`)

The `NormConfig` dataclass controls how rewards and advantages are normalized:

| Parameter        | Type        | Options                      | Description                                                |
| ---------------- | ----------- | ---------------------------- | ---------------------------------------------------------- |
| `mean_level`     | str \| None | `"batch"`, `"group"`, `None` | Level at which to compute mean for centering               |
| `std_level`      | str \| None | `"batch"`, `"group"`, `None` | Level at which to compute std for scaling                  |
| `mean_leave1out` | bool        | `true`, `false`              | Use leave-one-out average (exclude current sample)         |
| `std_unbiased`   | bool        | `true`, `false`              | Use unbiased std computation (default: `true`)             |
| `eps`            | float       | -                            | Small constant to avoid division by zero (default: `1e-5`) |
| `group_size`     | int         | -                            | Group size for group-level normalization                   |

"Batch" level computes the mean/std across the global batch, while "group" level
computes them within groups (e.g., trajectories sharing the same prompt). For
group-level normalization, `group_size` must be specified. Setting `mean_level` or
`std_level` to `None` skips mean subtraction or standard deviation scaling,
respectively.

If the entire field is omitted (e.g., `adv_norm: null` in YAML), no normalization is
performed.

Example:

```yaml
actor:
  adv_norm: null
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
```

**AReaL Default Practice**: The default configuration uses `std_level: batch` for
advantage normalization. This has been the AReaL team's standard practice across diverse
RL applications, from game AI (StarCraft) to LLM training (RLHF, reasoning, agentic
settings). While [Dr.GRPO](https://arxiv.org/abs/2503.20783) recommends
`std_level: null` for potentially improved performance, we retain `std_level: batch` for
backward compatibility. Users seeking Dr.GRPO-style behavior should set
`actor.adv_norm.std_level=null`.

### Clipping Strategy (`actor.eps_clip*`)

| Parameter         | Type          | Default | Description                                                                        |
| ----------------- | ------------- | ------- | ---------------------------------------------------------------------------------- |
| `eps_clip`        | float         | `0.2`   | Lower clipping bound: ratio clipped to `[1-eps_clip, ...]`                         |
| `eps_clip_higher` | float \| None | `None`  | Upper clipping bound: when set, ratio clipped to `[1-eps_clip, 1+eps_clip_higher]` |

When `eps_clip_higher` is `None`, symmetric clipping is used: $\text{clip}(r,
1-\epsilon, 1+\epsilon)$.

When `eps_clip_higher` is set (DAPO-style), asymmetric clipping is used:
$\text{clip}(r, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$.

### Importance Sampling Level (`actor.importance_sampling_level`)

| Parameter                   | Type | Options                 | Description                                 |
| --------------------------- | ---- | ----------------------- | ------------------------------------------- |
| `importance_sampling_level` | str  | `"token"`, `"sequence"` | Level at which to compute importance ratios |

- `"token"` (default): Standard per-token importance ratios (GRPO, PPO, etc.)
- `"sequence"` (GSPO): Sequence-level geometric mean of per-token ratios

## Algorithm Configuration Matrix

The following table shows how to configure each algorithm by setting the appropriate
parameters:

| Algorithm   | `adv_norm.mean_level` | `adv_norm.std_level` | `adv_norm.mean_leave1out` | `importance_sampling_level` | Special                           |
| ----------- | --------------------- | -------------------- | ------------------------- | --------------------------- | --------------------------------- |
| **PPO**     | `batch`               | `batch`              | `false`                   | `token`                     | critic model.                     |
| **GRPO**    | `batch`               | `batch`              | `false`                   | `token`                     | -                                 |
| **Dr.GRPO** | `group`               | `null`               | `false`                   | `token`                     | -                                 |
| **LitePPO** | `group`               | `batch`              | `false`                   | `token`                     | -                                 |
| **RLOO**    | `group`               | `null`               | `true`                    | `token`                     | -                                 |
| **GSPO**    | `batch`               | `batch`              | `false`                   | `sequence`                  | -                                 |
| **DAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | asymmetric clip, dynamic sampling |
| **SAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | `use_sapo_loss=true`              |

**Note**: The "GRPO" row reflects the original DeepSeekMath formulation. AReaL's default
GRPO config uses these settings but with length normalization already removed (see AReaL
Implementation Notes below).

## Algorithm-Specific Options

### Vanilla PPO

Vanilla PPO uses a learned value function (critic) to estimate advantages via GAE. The
key configuration difference is that it requires a `critic:` configuration section with
its own model and optimizer.

See `examples/math/gsm8k_ppo.yaml` for a complete configuration example.

### GRPO

$$ J_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q), \\ {o_i}_{i=1}^G
\sim \pi_{\theta_{\text{old}}}(O \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^G
\sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}\left(
r_{i,t}(\theta), 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) - \beta
D_{\mathrm{KL}}\left[ \pi_\theta \middle| \pi_{\text{ref}} \right] \right] $$

where:

$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, \quad
\hat{A}_{i,t} = \frac{r_i - \text{mean}({r_i}_{i=1}^G)}{\text{std}({r_i}_{i=1}^G)}.
$$

### RLOO (REINFORCE Leave-One-Out)

RLOO estimates the baseline by averaging rewards of **other** sampled responses
(excluding the current one). This is achieved by setting
`actor.adv_norm.mean_leave1out=true`.

$$ J_{\text{RLOO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q), \\ {o_i}_{i=1}^G
\sim \pi_{\theta_{\text{old}}}(O \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^G
\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},
\text{clip}\left( r_{i,t}(\theta), 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t}
\right) \right] $$

where:

$$ \hat{A}_{i,t} = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j. $$

### GSPO (Group Sequence Policy Optimization)

GSPO computes importance sampling ratios at the sequence level rather than the token
level.

**Standard PPO (token-level):**

$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} $$

**GSPO (sequence-level):**

$$ r_i(\theta) = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\log\frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}\right) $$

### SAPO (Soft Adaptive Policy Optimization)

SAPO replaces PPO's hard clipping with soft sigmoid gates, providing smooth gradients
and asymmetric control.

**Standard PPO:**

$$ L^{\text{PPO}} = -\mathbb{E}_t[\min(r_t A_t, r_t^{\text{clip}} A_t)] $$

**SAPO (with soft gates):**

- For positive advantages: $g_t^+ = \frac{4}{\tau_{\text{pos}}}
  \sigma(\tau_{\text{pos}} (r_t - 1))$
- For negative advantages: $g_t^- = \frac{4}{\tau_{\text{neg}}}
  \sigma(\tau_{\text{neg}} (r_t - 1))$
- Loss: $L^{\text{SAPO}} = -\mathbb{E}_t[g_t A_t]$ where $g_t = g_t^+$ if $A_t >
  0$, else $g_t^-$

| Parameter             | Type  | Default | Description                              |
| --------------------- | ----- | ------- | ---------------------------------------- |
| `actor.use_sapo_loss` | bool  | `false` | Enable SAPO loss instead of PPO clipping |
| `actor.sapo_tau_pos`  | float | `1.0`   | Temperature for positive advantages      |
| `actor.sapo_tau_neg`  | float | `1.05`  | Temperature for negative advantages      |

**Note:** SAPO requires `actor.use_decoupled_loss=false`.

```yaml
actor:
  use_sapo_loss: true
  sapo_tau_pos: 1.0
  sapo_tau_neg: 1.05
  use_decoupled_loss: false
```

### DAPO

DAPO introduces asymmetric clipping and dynamic sampling, which excludes samples where
all responses are uniformly correct or incorrect.

$$ J_{\text{DAPO}}(\theta) = \mathbb{E}_{\substack{(q,a) \sim \mathcal{D}, \\
{o_i}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(o \mid q)}} \left[
\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(
r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}\left( r_{i,t}(\theta),
1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}} \right) \hat{A}_{i,t}
\right) \right] $$

where $\hat{A}_{i,t}$ is the group-normalized advantage and $r_{i,t}(\theta)$ is the
token-level policy ratio.

**Asymmetric clipping parameters:**

| Parameter               | Type  | Default | Description                                     |
| ----------------------- | ----- | ------- | ----------------------------------------------- |
| `actor.eps_clip`        | float | `0.2`   | Lower clipping bound                            |
| `actor.eps_clip_higher` | float | -       | Upper clipping bound (set to enable asymmetric) |

**Overlong penalty parameters:**

| Parameter                       | Type  | Default | Description                                  |
| ------------------------------- | ----- | ------- | -------------------------------------------- |
| `actor.overlong_reward_penalty` | bool  | `false` | Enable penalty for overlong responses        |
| `actor.overlong_tokens`         | int   | -       | Number of tail tokens considered overlong    |
| `actor.overlong_penalty_factor` | float | -       | Penalty factor applied to overlong responses |

**Dynamic sampling:**

AReaL supports dynamic sampling via a `dynamic_filter_fn` passed to
`PPOTrainer.train()`. This function receives grouped trajectories sampled from the same
prompt and returns a boolean indicating whether to accept them for training:

```python
trainer.train(
    workflow=...,
    dynamic_filter_fn=lambda x: 0 < x["rewards"].mean() < 1
)
```

By default, AReaL uses a fixed batch size with dynamic filtering—it waits until
`batch_size` accepted samples are collected before training. This differs from some DAPO
implementations that use dynamic batch sizing, which collect an entire batch of samples
and then filter them. The following option controls batch sizing behavior:

| Parameter    | Type | Default | Description                 |
| ------------ | ---- | ------- | --------------------------- |
| `dynamic_bs` | bool | `false` | Enable dynamic batch sizing |

## Core Concepts

**Rewards**: AReaL assumes outcome-based rewards. Each trajectory, which may consist of
concatenated LLM input-output pairs, is assigned a single scalar reward at the sequence
level rather than at the token level.

**Advantages**: AReaL computes per-token advantages for each output token in the
trajectory. The PPO algorithm treats the outcome reward as the reward for the last
token, with all preceding tokens receiving a reward of 0. AReaL then applies standard
discounting and TD-error back-propagation via Generalized Advantage Estimation (GAE)
along the token trajectory to compute the advantage of each token. When the discount
factor is 1, the advantage values equal the outcome reward and are effectively broadcast
to every token in the trajectory.

## AReaL Implementation Notes

AReaL's GRPO implementation differs from the original DeepSeekMath paper in two key
ways:

**Length Normalization**: AReaL removes the per-token length normalization term from the
original GRPO objective. This aligns with recommendations from
[Dr.GRPO](https://arxiv.org/abs/2503.20783) and eliminates bias in advantage estimation.

**KL Regularization**: Instead of adding a KL divergence term directly to the objective
function, AReaL incorporates KL regularization into the advantage estimation
(PPO-style). The KL penalty is computed via `KLEstimator` and added to the per-token
rewards before GAE computation, controlled by the `actor.kl_ctl` parameter.
