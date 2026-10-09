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
- **IcePop**：[博客](https://ringtech.notion.site/icepop) — 基于重要性比率的token掩码（可与其它RL算法组合使用）
- **KPop**：[博客](https://ringtech.notion.site/kpop) — 双向二元KL散度token掩码（可与其它RL算法组合使用）

IcePop和KPop是token掩码策略，可以与上面列出的任意RL算法组合使用。

这些算法共享相同的基础目标，但在归一化策略、裁剪机制、重要性采样级别等方面有所不同。通过调整AReaL中的少量配置参数，你可以在不同算法之间切换。

## 示例用法

所有算法使用相同的执行模式。我们建议修改配置YAML文件中的参数。

| 后端      | 命令                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------- |
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=local` |
| **ray**   | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=ray`   |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_<algo>.yaml scheduler.type=slurm` |

将 `<algo>` 替换为：`ppo`、`grpo`、`drgrpo`、`liteppo`、`rloo`、`gspo`、`dapo_dynamic_bs`、`sapo`、`icepop` 或 `kpop`。

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

"Batch"级在整个全局批次上计算均值/标准差，而"group"级在组内计算（例如，共享相同提示的轨迹）。分组边界来自 rollout
批次元数据（`TrajBatchMeta.traj_group_sizes`）而非
`group_size`，因此即使各组大小不一（例如部分样本被过滤），仍会按提示逐组归一化；仅当该元数据不可用时，`group_size` 才作为固定步长的回退方案生效。将
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
| **DAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | 非对称裁剪，动态采样              |
| **SAPO**    | `batch`               | `batch`              | `false`                   | `token`                     | `use_sapo_loss=true`               |
| **IcePop**  | `batch`               | `batch`              | `false`                   | `token`                     | `rejection_sampling.metric=ratio`  |
| **KPop**    | `batch`               | `batch`              | `false`                   | `token`                     | `rejection_sampling.metric=binary_kl` |

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

### IcePop

IcePop对重要性比率 $r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ 超出可配置范围 $[\alpha, \beta]$ 的token进行掩码（其中 $\pi_\theta$ 为当前训练策略，$\pi_{\theta_\text{old}}$ 为采样时的行为策略）。重要性比率过低或过高的token不参与损失计算。

通过 `rejection_sampling` 配置的 `metric=ratio` 实现：

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

| 参数                                 | 类型        | 默认值 | 描述                       |
| ------------------------------------ | ----------- | ------ | -------------------------- |
| `actor.rejection_sampling.metric`    | str         | -      | 设置为 `ratio` 以启用IcePop |
| `actor.rejection_sampling.lower`     | float       | `0.5`  | 重要性比率下界             |
| `actor.rejection_sampling.upper`     | float       | `5.0`  | 重要性比率上界             |

**注意：** IcePop需要 `actor.use_decoupled_loss=true`，否则 `rejection_sampling` 不生效。

完整的配置示例见 `examples/math/gsm8k_icepop.yaml`。

### KPop

KPop对双向二元KL散度超过阈值的token进行掩码。对于每个token，计算：

$$\text{KL}_{\text{fwd}} = \text{KL}(P_\theta \| P_{\theta_\text{old}}), \quad \text{KL}_{\text{rev}} = \text{KL}(P_{\theta_\text{old}} \| P_\theta)$$

其中每个token概率被视为伯努利参数：$\text{KL}(P \| Q) = p \log \frac{p}{q} + (1-p) \log \frac{1-p}{1-q}$。$\max(\text{KL}_{\text{fwd}}, \text{KL}_{\text{rev}}) > \phi$ 的token被掩码（此处 $\phi$ 对应 `actor.rejection_sampling.upper`）。

通过 `rejection_sampling` 配置的 `metric=binary_kl` 实现：

```yaml
actor:
  use_decoupled_loss: true
  rejection_sampling:
    level: token
    action: mask
    metric: binary_kl
    upper: 2.0
```

| 参数                                 | 类型        | 默认值 | 描述                           |
| ------------------------------------ | ----------- | ------ | ------------------------------ |
| `actor.rejection_sampling.metric`    | str         | -      | 设置为 `binary_kl` 以启用KPop  |
| `actor.rejection_sampling.upper`     | float       | `2.0`  | KL散度阈值（$\phi$）          |

**注意：** KPop仅支持 `action=mask`（不支持 `clamp`），且 `lower` 在 `binary_kl` 中不使用。KPop需要 `actor.use_decoupled_loss=true`，否则 `rejection_sampling` 不生效。

完整的配置示例见 `examples/math/gsm8k_kpop.yaml`。

## 核心概念

**奖励**：AReaL假设基于结果的奖励。每个可能由连接的LLM输入-输出对组成的轨迹，在序列级而非token级被分配一个标量奖励。

**优势**：AReaL为轨迹中的每个输出token计算逐token优势。PPO算法将结果奖励视为最后一个token的奖励，所有前面的token奖励为0。然后AReaL通过广义优势估计（GAE）应用标准折扣和TD误差反向传播来计算每个token的优势。使用默认的token级递推时，如果`discount=1`、`gae_lambda=1`、critic value为0且禁用KL正则化，终止结果奖励会等效广播到每个生成token。

### GAE时间步单位

`actor.gae_timestep_unit`用于选择GAE按生成token还是生成turn推进。Prompt、tool、padding及其他被mask的位置都不会消耗GAE时间步。

#### Token级GAE

`token`是默认值，并保留AReaL原有行为。对于相邻的有效生成token，AReaL计算：

$$
\delta_t = r_t + \gamma V_{t+1} - V_t, \qquad
A_t = \delta_t + \gamma \lambda A_{t+1}, \qquad
G_t = A_t + V_t,
$$

其中$\gamma$为`actor.discount`，$\lambda$为`actor.gae_lambda`配置解析出的逐轨迹取值。奖励$r_t$包含结果奖励增量和token级KL惩罚。以EOS结束的轨迹使用0作为终止bootstrap；没有EOS的截断轨迹则使用最后一个value估计进行bootstrap。

#### Turn级GAE

`turn`将每个非空生成turn视为一个宏观时间步。对于turn $u$，AReaL将其中的task reward增量求和为$r_u^{\mathrm{task}}$，并使用该turn第一个有效action token位置的value作为$V_u$，然后计算：

$$
\delta_u^{\mathrm{task}}
= r_u^{\mathrm{task}} + \gamma V_{u+1} - V_u, \qquad
A_u^{\mathrm{task}}
= \delta_u^{\mathrm{task}} + \gamma \lambda A_{u+1}^{\mathrm{task}}.
$$

Task advantage $A_u^{\mathrm{task}}$和critic target $G_u=A_u^{\mathrm{task}}+V_u$会广播到该turn中的每个有效token。Token级KL不会进入turn递推和critic target；在可选的`actor.adv_norm`之前，turn $u$中token $j$的actor advantage为$A_{u,j}=A_u^{\mathrm{task}}+r_{u,j}^{\mathrm{KL}}$。这样可以避免先对一个turn的KL惩罚求和，再将总和广播回每个token。

### 动态GAE lambda

`actor.gae_lambda`既可以是静态float，也可以是callable的点分路径。`actor.gae_lambda_kwargs`用于向callable传递关键字参数，配置静态float时会被忽略。该函数接收包含三个`[B]`形状tensor的context：

- `effective_token_lengths`：有效生成token数量，包括有效的EOS token；
- `turn_counts`：非空生成turn数量；token模式缺少`turn_ids`时为0；
- `timestep_lengths`：由`gae_timestep_unit`选择的长度$L$。

Callable必须在相同device上返回`[B]`形状的有限浮点tensor，即每条本地轨迹一个lambda。返回的lambda会用于该轨迹的所有选定时间步。

内置了两个长度自适应函数：

| 函数路径 | 参数 | 定义 |
| --- | --- | --- |
| `areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae` | `alpha > 0` | $L>0$时$\lambda=\max(0, 1 - 1/(\alpha L))$；$L=0$时取0。 |
| `areal.trainer.ppo.lambda_fn.relative_position_gae_lambda` | `0 < q <= 1` | $L\ge2$时$\lambda=q^{1/(L-1)}$；$L=1$时取1，$L=0$时取0。相对保留率解释假设$\gamma=1$。 |

配置示例：

```yaml
actor:
  gae_timestep_unit: turn
  gae_lambda: areal.trainer.ppo.lambda_fn.relative_position_gae_lambda
  gae_lambda_kwargs:
    q: 0.5
```

### 自定义workflow的`turn_ids`约定

Turn级GAE要求workflow返回原始、与token对齐的`turn_ids`；actor会在内部将其与next-token prediction mask对齐。该tensor必须满足：

- 与`input_ids`和`loss_mask`形状相同（batch后为`[B, S]`）；
- 使用整数dtype（建议使用有符号整数来表示`-1`哨兵值）；
- 为每个有效生成token分配`[0, S)`范围内的ID；
- 有效ID随时间单调不减，同一个assistant turn中的所有token使用相同ID；
- prompt、user、tool、padding及其他非loss位置使用`-1`。

编号允许跳跃且不会额外消耗GAE时间步，但建议从0开始连续编号。自定义workflow可以按如下方式构造该字段；不要在workflow中进行roll：

```python
turn_ids += [-1] * input_len + [turn_idx] * resp.output_len
result["turn_ids"] = torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0)
```

## AReaL实现说明

AReaL的GRPO实现在两个关键方面与原始DeepSeekMath论文不同：

**长度归一化**：AReaL从原始GRPO目标中移除了逐token长度归一化项。这与 [Dr.GRPO](https://arxiv.org/abs/2503.20783)
的建议一致，并消除了优势估计中的偏差。

**KL正则化**：AReaL不是将KL散度项直接添加到目标函数中，而是通过`actor.kl_ctl`将KL正则化纳入actor advantage（PPO风格）。在token模式下，`KLEstimator`计算的惩罚会在GAE之前加入逐token奖励；在turn模式下，它仅作为token局部的actor惩罚，不进入turn递推和critic target。
