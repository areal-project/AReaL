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
- **IcePop**: [Blog](https://ringtech.notion.site/icepop) — Importance-ratio-based token masking (composable with other RL algorithms)
- **KPop**: [Blog](https://ringtech.notion.site/kpop) — Bidirectional binary KL divergence token masking (composable with other RL algorithms)

IcePop and KPop are token masking strategies that can be composed with any RL algorithm
listed above.

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
`dapo_dynamic_bs`, `sapo`, `icepop`, or `kpop`.

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
computes them within groups (e.g., trajectories sharing the same prompt). Group
boundaries come from the rollout batch metadata (`TrajBatchMeta.traj_group_sizes`)
rather than from `group_size`, so groups of unequal size (e.g., when some samples are
filtered out) are still normalized per prompt; `group_size` applies only as a
fixed-stride fallback when that metadata is unavailable. Setting `mean_level` or
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
| **DAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | asymmetric clip, dynamic sampling      |
| **SAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | `use_sapo_loss=true`                   |
| **IcePop**  | `batch`               | `batch`              | `false`                   | `token`                     | `rejection_sampling.metric=ratio`      |
| **KPop**    | `batch`               | `batch`              | `false`                   | `token`                     | `rejection_sampling.metric=binary_kl`  |

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

### IcePop

IcePop masks tokens whose importance ratio $r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ falls outside a configurable range $[\alpha, \beta]$ (where $\pi_\theta$ is the current training policy and $\pi_{\theta_\text{old}}$ is the behavior policy used for rollout). Tokens with too-low or too-high importance ratios are excluded from the loss.

It is implemented via the `rejection_sampling` config with `metric=ratio`:

```yaml
actor:
  use_decoupled_loss: true
  rejection_sampling:
    level: token
    action: mask
    metric: ratio
    lower: 0.5
    upper: 5.0
```

| Parameter                            | Type        | Default | Description                     |
| ------------------------------------ | ----------- | ------- | ------------------------------- |
| `actor.rejection_sampling.metric`    | str         | -       | Set to `ratio` for IcePop       |
| `actor.rejection_sampling.lower`     | float       | `0.5`   | Lower bound of importance ratio |
| `actor.rejection_sampling.upper`     | float       | `5.0`   | Upper bound of importance ratio |

**Note:** IcePop requires `actor.use_decoupled_loss=true`, otherwise `rejection_sampling` has no effect.

See `examples/math/gsm8k_icepop.yaml` for a complete configuration example.

### KPop

KPop masks tokens where the bidirectional binary KL divergence exceeds a threshold. For each token, it computes:

$$\text{KL}_{\text{fwd}} = \text{KL}(P_\theta \| P_{\theta_\text{old}}), \quad \text{KL}_{\text{rev}} = \text{KL}(P_{\theta_\text{old}} \| P_\theta)$$

where each token probability is treated as a Bernoulli parameter: $\text{KL}(P \| Q) = p \log \frac{p}{q} + (1-p) \log \frac{1-p}{1-q}$. Tokens where $\max(\text{KL}_{\text{fwd}}, \text{KL}_{\text{rev}}) > \phi$ are masked (here $\phi$ corresponds to `actor.rejection_sampling.upper`).

It is implemented via the `rejection_sampling` config with `metric=binary_kl`:

```yaml
actor:
  use_decoupled_loss: true
  rejection_sampling:
    level: token
    action: mask
    metric: binary_kl
    upper: 2.0
```

| Parameter                            | Type        | Default | Description                          |
| ------------------------------------ | ----------- | ------- | ------------------------------------ |
| `actor.rejection_sampling.metric`    | str         | -       | Set to `binary_kl` for KPop          |
| `actor.rejection_sampling.upper`     | float       | `2.0`   | KL divergence threshold ($\phi$)     |

**Note:** KPop only supports `action=mask` (not `clamp`), and `lower` is not used with `binary_kl`. KPop requires `actor.use_decoupled_loss=true`, otherwise `rejection_sampling` has no effect.

See `examples/math/gsm8k_kpop.yaml` for a complete configuration example.

## Core Concepts

**Rewards**: AReaL assumes outcome-based rewards. Each trajectory, which may consist of
concatenated LLM input-output pairs, is assigned a single scalar reward at the sequence
level rather than at the token level.

**Advantages**: AReaL computes per-token advantages for each output token in the
trajectory. The PPO algorithm treats the outcome reward as the reward for the last
token, with all preceding tokens receiving a reward of 0. AReaL then applies standard
discounting and TD-error back-propagation via Generalized Advantage Estimation (GAE)
to compute the advantage of each token. With the default token-level recurrence, the
terminal outcome reward is effectively broadcast to every generated token when
`discount=1`, `gae_lambda=1`, critic values are zero, and KL regularization is disabled.

### GAE timestep units

`actor.gae_timestep_unit` selects whether GAE advances over generated tokens or
generated turns. Prompt, tool, padding, and other masked positions never consume a GAE
step.

#### Token-level GAE

`token` is the default and preserves the original AReaL behavior. For consecutive active
generated tokens, AReaL computes

$$
\delta_t = r_t + \gamma V_{t+1} - V_t, \qquad
A_t = \delta_t + \gamma \lambda A_{t+1}, \qquad
G_t = A_t + V_t,
$$

where $\gamma$ is `actor.discount` and $\lambda$ is the resolved per-trajectory value
configured by `actor.gae_lambda`. The reward $r_t$ contains the outcome reward increment
and the token-level KL penalty. EOS-terminated trajectories use zero terminal bootstrap,
while truncated trajectories without EOS bootstrap from the final value estimate.

#### Turn-level GAE

`turn` treats each non-empty generated turn as one macro timestep. For turn $u$, AReaL
sums its task reward increments into $r_u^{\mathrm{task}}$ and takes $V_u$ from the first
active action-token position in that turn. It then computes

$$
\delta_u^{\mathrm{task}}
= r_u^{\mathrm{task}} + \gamma V_{u+1} - V_u, \qquad
A_u^{\mathrm{task}}
= \delta_u^{\mathrm{task}} + \gamma \lambda A_{u+1}^{\mathrm{task}}.
$$

The task advantage $A_u^{\mathrm{task}}$ and critic target
$G_u=A_u^{\mathrm{task}}+V_u$ are broadcast to every active token in the turn.
Token-level KL is deliberately kept out of the turn recurrence and critic target: before
optional `actor.adv_norm`, the actor advantage for token $j$ in turn $u$ is
$A_{u,j}=A_u^{\mathrm{task}}+r_{u,j}^{\mathrm{KL}}$. This avoids summing a turn's KL
penalties and then broadcasting that sum back to every token.

### Dynamic GAE lambda

`actor.gae_lambda` accepts either a static float or a dotted path to a callable.
`actor.gae_lambda_kwargs` passes keyword arguments to that callable and is ignored for a
static float. The callable receives a context containing three tensors of shape `[B]`:

- `effective_token_lengths`: active generated-token counts, including an active EOS;
- `turn_counts`: non-empty generated-turn counts, or zeros when token mode has no
  `turn_ids`;
- `timestep_lengths`: the length $L$ selected by `gae_timestep_unit`.

The callable must return one finite floating-point lambda per local trajectory as a
tensor of shape `[B]` on the same device. A returned lambda is used for every selected
timestep in that trajectory.

Two length-aware functions are built in:

| Function path | Kwargs | Definition |
| --- | --- | --- |
| `areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae` | `alpha > 0` | $\lambda=\max(0, 1 - 1/(\alpha L))$ for $L>0$; $L=0$ uses 0. |
| `areal.trainer.ppo.lambda_fn.relative_position_gae_lambda` | `0 < q <= 1` | $\lambda=q^{1/(L-1)}$ for $L\ge2$; $L=1$ uses 1 and $L=0$ uses 0. Relative-retention interpretation assumes $\gamma=1$. |

For example:

```yaml
actor:
  gae_timestep_unit: turn
  gae_lambda: areal.trainer.ppo.lambda_fn.relative_position_gae_lambda
  gae_lambda_kwargs:
    q: 0.5
```

### `turn_ids` contract for custom workflows

Turn-level GAE requires workflows to return raw, token-aligned `turn_ids`; the actor
aligns them with its next-token prediction mask internally. The tensor must:

- have the same shape as `input_ids` and `loss_mask` (batched as `[B, S]`);
- use an integer dtype (a signed integer is recommended for the `-1` sentinel);
- assign every active generated token an ID in `[0, S)`;
- keep active IDs temporally nondecreasing and use one ID for all tokens in the same
  assistant turn;
- use `-1` for prompt, user, tool, padding, and other non-loss positions.

Numbering gaps are accepted and do not consume a GAE step, although consecutive IDs
starting from zero are recommended. A custom workflow can construct the field as
follows; do not roll it in the workflow:

```python
turn_ids += [-1] * input_len + [turn_idx] * resp.output_len
result["turn_ids"] = torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0)
```

## AReaL Implementation Notes

AReaL's GRPO implementation differs from the original DeepSeekMath paper in two key
ways:

**Length Normalization**: AReaL removes the per-token length normalization term from the
original GRPO objective. This aligns with recommendations from
[Dr.GRPO](https://arxiv.org/abs/2503.20783) and eliminates bias in advantage estimation.

**KL Regularization**: Instead of adding a KL divergence term directly to the objective
function, AReaL incorporates KL regularization into actor advantages (PPO-style),
controlled by `actor.kl_ctl`. In token mode, the `KLEstimator` penalty is added to
per-token rewards before GAE. In turn mode, it remains a token-local actor penalty and
is excluded from the turn recurrence and critic targets.
