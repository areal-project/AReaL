# 异步强化学习

AReaL原生支持异步强化学习训练，使推理和训练能够在不同的GPU上并行进行，从而最大化GPU利用率。

> **注意：** 本指南适用于启用异步训练时的所有算法（即 `rollout.max_head_offpolicyness > 0`）。设置
> `rollout.max_head_offpolicyness=0` 会将AReaL恢复为同步RL。同步设置对调试很有用，但通常比异步训练慢2倍。

## 概述

传统的在线强化学习算法假设同步执行：模型生成rollout，在其上训练，然后重复。虽然简单，但这种方法在长rollout期间会让GPU处于空闲状态，并且扩展性不佳。

异步强化学习通过重叠rollout生成和训练来打破这一限制。然而，这引入了**离策略（off-policyness）**：生成rollout的策略版本可能落后于训练版本。为了最大化推理吞吐量，AReaL还支持**部分rollout**，即单个轨迹可以跨多个策略版本进行分段。

## 关键技术

AReaL使用两种互补技术来解决上述算法挑战：

### 1. 离策略控制

限制过时的rollout相对于当前训练策略的程度：

```yaml
rollout:
  max_head_offpolicyness: 4  # 允许落后最多4个版本步骤
```

**配置建议：**

- 设置为 `0` 用于同步RL（适用于调试或基线比较）
- 更高的值会增加吞吐量，但可能会降低训练稳定性
- 典型范围：2-8，取决于模型大小和更新频率

### 2. 解耦PPO目标

使用修改后的损失计算来处理离策略数据：

```yaml
actor:
  use_decoupled_loss: true     # 启用解耦PPO目标
  recompute_logprobs: true     # 在训练期间重新计算logprobs
```

**配置选项：**

- `use_decoupled_loss`：设为 `false` 时，使用标准PPO/GRPO目标
- `recompute_logprobs`：设为 `false` 时，重用推理后端的logprobs
  - **注意：** 当启用 `use_decoupled_loss` 时必须设为 `true`

> **注意：** 解耦PPO损失可能与某些算法配置（例如SAPO）冲突。异步性对这些较新算法的影响在很大程度上尚未得到研究。

## 参考资料

有关异步训练的实践教程，请参阅我们的 [GSM8K GRPO示例](../tutorial/gsm8k_grpo.md)。

有关算法细节和实证分析，请参阅 [AReaL论文](https://arxiv.org/pdf/2505.24298)。
