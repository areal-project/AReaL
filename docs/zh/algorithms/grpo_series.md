# PPO、GRPO及相关算法

最后更新：2026年1月4日

作者： [Ziyi ZENG](https://github.com/ZiyiTsang),
[Wei Fu](https://github.com/garrett4wade), [Honghua DONG](https://github.com/dhh1995),
[Bruce Wu](https://github.com/Bruce-rl-hw), [Bruce Li](https://github.com/HsiaoTsan)

本文档涵盖了一系列用于LLM训练的类PPO强化学习算法，包括：

- **Vanilla PPO**
- **GRPO** (DeepSeekMath): [论文](https://arxiv.org/pdf/2402.03300)
- **Dr.GRPO**: [论文](https://arxiv.org/abs/2503.20783)
- **LitePPO**: [论文](https://arxiv.org/pdf/2508.08221v1)
- **RLOO**: [论文](https://arxiv.org/abs/2402.14740)
- **DAPO**: [论文](https://arxiv.org/abs/2503.14476)
- **SAPO**: [论文](https://arxiv.org/abs/2511.20347)
- **GSPO** (Qwen3): [论文](https://arxiv.org/abs/2507.18071)，
  [博客](https://qwenlm.github.io/blog/gspo/)

这些算法共享相同的基础目标，但在归一化策略、裁剪机制、重要性采样级别等方面有所不同。通过调整AReaL中的少量配置参数，你可以在不同算法之间切换。

## 示例用法

所有算法使用相同的执行模式。我们建议修改配置YAML文件中的参数。

| 后端      | 命令                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------- |
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=local` |
| **ray**   | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=ray`   |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=slurm` |

将 `<algo>` 替换为：`ppo`、`grpo`、`drgrpo`、`liteppo`、`rloo`、`gspo`、`dapo_dynamic_bs` 或 `sapo`。

### 通过CLI覆盖切换算法

你也可以通过覆盖配置参数来切换算法：

```bash
# Dr.GRPO (从GRPO配置)
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  actor.adv_norm.mean_level=group \
  actor.adv_norm.std_level=null

# GSPO (从GRPO配置)
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  +actor.importance_sampling_level=sequence

# SAPO (从GRPO配置)
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  +actor.use_sapo_loss=true \
  +actor.sapo_tau_pos=1.0 \
  +actor.sapo_tau_neg=1.05 \
  actor.use_decoupled_loss=false
```

注意：添加原始YAML中不存在的键时请使用 `+` 前缀。

## 核心配置参数

所有配置都定义在 `areal/api/cli_args.py` 中的 `PPOActorConfig` 和 `NormConfig` 下。详见
[CLI配置](../cli_reference.md)。

### 奖励和优势归一化（`actor.reward_norm` 和 `actor.adv_norm`）

`NormConfig` 数据类控制奖励和优势的归一化方式：

| 参数             | 类型        | 选项                         | 描述                               |
| ---------------- | ----------- | ---------------------------- | ---------------------------------- |
| `mean_level`     | str \| None | `"batch"`、`"group"`、`None` | 计算均值的级别                     |
| `std_level`      | str \| None | `"batch"`、`"group"`、`None` | 计算标准差的级别                   |
| `mean_leave1out` | bool        | `true`、`false`              | 使用留一法平均值（排除当前样本）   |
| `std_unbiased`   | bool        | `true`、`false`              | 使用无偏标准差计算（默认：`true`） |
| `eps`            | float       | -                            | 避免除零的小常数（默认：`1e-5`）   |
| `group_size`     | int         | -                            | 分组级归一化的组大小               |

"Batch"级在整个全局批次上计算均值/标准差，而"group"级在组内计算（例如，共享相同提示的轨迹）。对于分组级归一化，必须指定 `group_size`。将
`mean_level` 或 `std_level` 设为 `None` 分别跳过均值减法或标准差缩放。

如果整个字段被省略（例如YAML中的 `adv_norm: null`），则不执行归一化。

示例：

```yaml
actor:
  adv_norm: null
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
```

**AReaL默认实践**：默认配置使用 `std_level: batch`
进行优势归一化。这已成为AReaL团队在各种RL应用中的标准实践，从游戏AI（StarCraft）到LLM训练（RLHF、推理、agent设置）。虽然
[Dr.GRPO](https://arxiv.org/abs/2503.20783) 建议使用 `std_level: null` 以获得潜在更好的性能，但我们保留
`std_level: batch` 以保持向后兼容性。寻求Dr.GRPO风格行为的用户应设置 `actor.adv_norm.std_level=null`。

### 裁剪策略（`actor.eps_clip*`）

| 参数              | 类型          | 默认值 | 描述                                                             |
| ----------------- | ------------- | ------ | ---------------------------------------------------------------- |
| `eps_clip`        | float         | `0.2`  | 下裁剪边界：比率裁剪到 `[1-eps_clip, ...]`                       |
| `eps_clip_higher` | float \| None | `None` | 上裁剪边界：设置时，比率裁剪到 `[1-eps_clip, 1+eps_clip_higher]` |

当 `eps_clip_higher` 为 `None` 时，使用对称裁剪： $\text{clip}(r, 1-\epsilon, 1+\epsilon)$。

当设置 `eps_clip_higher` 时（DAPO风格），使用非对称裁剪： $\text{clip}(r, 1-\epsilon_{\text{low}},
1+\epsilon_{\text{high}})$。

### 重要性采样级别（`actor.importance_sampling_level`）

| 参数                        | 类型 | 选项                    | 描述                 |
| --------------------------- | ---- | ----------------------- | -------------------- |
| `importance_sampling_level` | str  | `"token"`、`"sequence"` | 计算重要性比率的级别 |

- `"token"`（默认）：标准逐token重要性比率（GRPO、PPO等）
- `"sequence"`（GSPO）：逐token比率的序列级几何平均值

## 算法配置矩阵

下表展示了如何通过设置适当的参数来配置每个算法：

| 算法        | `adv_norm.mean_level` | `adv_norm.std_level` | `adv_norm.mean_leave1out` | `importance_sampling_level` | 特殊配置             |
| ----------- | --------------------- | -------------------- | ------------------------- | --------------------------- | -------------------- |
| **PPO**     | `batch`               | `batch`              | `false`                   | `token`                     | 需要critic模型。     |
| **GRPO**    | `batch`               | `batch`              | `false`                   | `token`                     | -                    |
| **Dr.GRPO** | `group`               | `null`               | `false`                   | `token`                     | -                    |
| **LitePPO** | `group`               | `batch`              | `false`                   | `token`                     | -                    |
| **RLOO**    | `group`               | `null`               | `true`                    | `token`                     | -                    |
| **GSPO**    | `batch`               | `batch`              | `false`                   | `sequence`                  | -                    |
| **DAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | 非对称裁剪，动态采样 |
| **SAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | `use_sapo_loss=true` |

**注意**："GRPO"行反映原始DeepSeekMath公式。AReaL的默认GRPO配置使用这些设置，但已移除长度归一化（见下文AReaL实现说明）。

## 算法特定选项

### Vanilla PPO

Vanilla PPO使用学习到的价值函数（critic）通过GAE估计优势。关键配置差异是它需要一个 `critic:` 配置部分，包含自己的模型和优化器。

完整的配置示例见 `examples/math/gsm8k_ppo.yaml`。

### GRPO

$$ J_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q), \\ {o_i}_{i=1}^G
\sim \pi_{\theta_{\text{old}}}(O \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^G
\sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}\left(
r_{i,t}(\theta), 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) - \beta
D_{\mathrm{KL}}\left[ \pi_\theta \middle| \pi_{\text{ref}} \right] \right] $$

其中：

$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, \quad
\hat{A}_{i,t} = \frac{r_i - \text{mean}({r_i}_{i=1}^G)}{\text{std}({r_i}_{i=1}^G)}.
$$

### RLOO (REINFORCE Leave-One-Out)

RLOO通过平均**其他**采样响应的奖励（排除当前响应）来估计基线。这通过设置 `actor.adv_norm.mean_leave1out=true` 实现。

$$ J_{\text{RLOO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q), \\ {o_i}_{i=1}^G
\sim \pi_{\theta_{\text{old}}}(O \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^G
\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},
\text{clip}\left( r_{i,t}(\theta), 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t}
\right) \right] $$

其中：

$$ \hat{A}_{i,t} = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j. $$

### GSPO (Group Sequence Policy Optimization)

GSPO在序列级别而非token级别计算重要性采样比率。

**标准PPO（token级）：**

$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} $$

**GSPO（序列级）：**

$$ r_i(\theta) = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\log\frac{\pi_\theta(o_{i,t} \mid q,
o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}\right) $$

### SAPO (Soft Adaptive Policy Optimization)

SAPO用软sigmoid门替换PPO的硬裁剪，提供平滑梯度和非对称控制。

**标准PPO：**

$$ L^{\text{PPO}} = -\mathbb{E}_t[\min(r_t A_t, r_t^{\text{clip}} A_t)] $$

**SAPO（带软门）：**

- 对于正向优势：$g_t^+ = \frac{4}{\tau_{\text{pos}}} \sigma(\tau_{\text{pos}} (r_t -
  1))$
- 对于负向优势：$g_t^- = \frac{4}{\tau_{\text{neg}}} \sigma(\tau_{\text{neg}} (r_t -
  1))$
- 损失：$L^{\text{SAPO}} = -\mathbb{E}_t[g_t A_t]$，其中如果 $A_t > 0$ 则 $g_t = g_t^+$，否则
  $g_t = g_t^-$

| 参数                  | 类型  | 默认值  | 描述                    |
| --------------------- | ----- | ------- | ----------------------- |
| `actor.use_sapo_loss` | bool  | `false` | 启用SAPO损失代替PPO裁剪 |
| `actor.sapo_tau_pos`  | float | `1.0`   | 正向优势的温度参数      |
| `actor.sapo_tau_neg`  | float | `1.05`  | 负向优势的温度参数      |

**注意：** SAPO需要 `actor.use_decoupled_loss=false`。

```yaml
actor:
  use_sapo_loss: true
  sapo_tau_pos: 1.0
  sapo_tau_neg: 1.05
  use_decoupled_loss: false
```

### DAPO

DAPO引入非对称裁剪和动态采样，后者排除所有响应都完全正确或完全错误的样本。

$$ J_{\text{DAPO}}(\theta) = \mathbb{E}_{\substack{(q,a) \sim \mathcal{D}, \\
{o_i}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(o \mid q)}} \left[
\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left(
r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}\left( r_{i,t}(\theta),
1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}} \right) \hat{A}_{i,t}
\right) \right] $$

其中 $\hat{A}_{i,t}$ 是分组归一化优势，$r_{i,t}(\theta)$ 是token级策略比率。

**非对称裁剪参数：**

| 参数                    | 类型  | 默认值 | 描述                           |
| ----------------------- | ----- | ------ | ------------------------------ |
| `actor.eps_clip`        | float | `0.2`  | 下裁剪边界                     |
| `actor.eps_clip_higher` | float | -      | 上裁剪边界（设置以启用非对称） |

**过长惩罚参数：**

| 参数                            | 类型  | 默认值  | 描述                      |
| ------------------------------- | ----- | ------- | ------------------------- |
| `actor.overlong_reward_penalty` | bool  | `false` | 启用过长响应惩罚          |
| `actor.overlong_tokens`         | int   | -       | 被视为过长的尾部token数量 |
| `actor.overlong_penalty_factor` | float | -       | 应用于过长响应的惩罚因子  |

**动态采样：**

AReaL通过传递给 `PPOTrainer.train()` 的 `dynamic_filter_fn`
支持动态采样。该函数接收从相同提示采样的分组轨迹，并返回布尔值指示是否接受它们进行训练：

```python
trainer.train(
    workflow=...,
    dynamic_filter_fn=lambda x: 0 < x["rewards"].mean() < 1
)
```

默认情况下，AReaL使用固定批量大小的动态过滤——它等待收集到 `batch_size`
个接受样本后再进行训练。这与某些使用动态批量大小的DAPO实现不同，后者收集整个批次的样本然后过滤它们。以下选项控制批量大小行为：

| 参数         | 类型 | 默认值  | 描述             |
| ------------ | ---- | ------- | ---------------- |
| `dynamic_bs` | bool | `false` | 启用动态批量大小 |

## 核心概念

**奖励**：AReaL假设基于结果的奖励。每个可能由连接的LLM输入-输出对组成的轨迹，在序列级而非token级被分配一个标量奖励。

**优势**：AReaL为轨迹中的每个输出token计算逐token优势。PPO算法将结果奖励视为最后一个token的奖励，所有前面的token奖励为0。然后AReaL通过沿token轨迹的广义优势估计（GAE）应用标准折扣和TD误差反向传播来计算每个token的优势。当折扣因子为1时，优势值等于结果奖励，并有效地广播到轨迹中的每个token。

## AReaL实现说明

AReaL的GRPO实现在两个关键方面与原始DeepSeekMath论文不同：

**长度归一化**：AReaL从原始GRPO目标中移除了逐token长度归一化项。这与 [Dr.GRPO](https://arxiv.org/abs/2503.20783)
的建议一致，并消除了优势估计中的偏差。

**KL正则化**：AReaL不是将KL散度项直接添加到目标函数中，而是将KL正则化纳入优势估计（PPO风格）。KL惩罚通过 `KLEstimator`
计算，并在GAE计算之前添加到逐token奖励中，由 `actor.kl_ctl` 参数控制。
