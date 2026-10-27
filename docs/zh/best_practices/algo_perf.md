# 诊断 RL 性能问题

本指南帮助您诊断和解决强化学习训练中的常见性能问题。使用以下策略来识别瓶颈、调整超参数并优化您的 RL Workflow。

## 使用同步 RL 而非异步训练

如果您怀疑异步 RL 训练影响了学习效果，或者想要调试新的 Agentic 应用，可以使用以下配置切换到标准的同步 RL 训练：

```yaml
rollout:
  max_head_offpolicyness: 0  # 0 表示同步训练
actor:
  recompute_logprob: false  # 使用推理后端返回的 logprobs
  use_decoupled_loss: false  # 恢复为原始的 PPO 损失
```

有关这些配置的详细信息，请参阅我们的[异步 RL 指南](../algorithms/async.md)和 [CLI 参考文档](../cli_reference.md)。

## 训练奖励不增加

这是一个常见问题，可能有多种原因。我们建议按以下步骤进行诊断：

1. **建立基线：** 在训练前运行测试集评估以测量基线性能。AReaL
   允许在训练和评估之间零代码更改，因此您可以重用训练代码进行评估。详细信息请参阅[评估指南](../tutorial/eval.md)。
1. **使用更简单的数据测试：** 在测试集而非训练集上运行 RL 训练，以验证奖励是否增加。
1. **如果在测试集上奖励不增加：** 调整超参数（例如增加批大小或学习率）或切换到其他基础模型。考虑先应用 SFT，因为这表明当前模型可能无法完成该任务。
1. **如果在测试集上奖励增加但训练集上不增加：** 检查训练数据的质量和难度。确保分布匹配且难度适合您的基础模型。您可以通过向 `prepare_batch` 传递
   `should_accept_fn` 参数来启用动态过滤（类似于
   DAPO），以确保任务难度在运行时保持适当。更多信息请参阅我们的[详细代码演练](../tutorial/gsm8k_grpo.md)。

## 需要监控的重要指标

监控这些指标有助于确保训练稳定并及早发现问题。

### 奖励指标

| 指标                    | 描述                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `eval-rollout/reward`   | **测试集**上的奖励。模型泛化的主要指标。                                                                                          |
| `rollout/reward`        | **训练集**上的奖励。跟踪训练期间的学习进度。                                                                                      |
| `ppo/actor/task_reward` | **实际用于训练**的轨迹的奖励。当启用动态过滤时，这与 `rollout/reward` 不同——过滤掉的轨迹在此处被排除，但仍计入 `rollout/reward`。 |

**排除 `task_reward` 高方差的故障：**

如果 `task_reward` 波动很大，您的训练具有高方差。如果资源允许，考虑增加批大小——这通常是一种有效的补救方法。

### 重要性权重指标

我们推荐使用**带解耦 PPO
损失**的异步训练（`use_decoupled_loss=true`）以获得最佳吞吐量。以下两个重要性权重指标对于监控训练稳定性至关重要——它们的平均值应保持接近 1.0。

使用 `use_decoupled_loss=true` 时，损失函数分离三个策略：

- **π_behave**：在 rollout 期间生成样本的行为策略
- **π_proximal**：近端策略，比当前策略晚训练一步
- **π_θ**：正在优化的当前策略

解耦 PPO 损失结合两个重要性权重：

$$L = -\mathbb{E}\left[
\underbrace{\frac{\pi_{\text{proximal}}}{\pi_{\text{behave}}}}_{\text{behave imp weight}}
\cdot \min\left(
\underbrace{\frac{\pi_\theta}{\pi_{\text{proximal}}}}_{\text{importance weight}}
A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{proximal}}}, 1-\epsilon,
1+\epsilon\right) A \right) \right]$$

| 指标                                 | 公式                  | 描述                                  |
| ------------------------------------ | --------------------- | ------------------------------------- |
| `ppo_actor/update/importance_weight` | π_θ / π_proximal      | 当前策略与近端策略之间 PPO 剪裁的比率 |
| `ppo_actor/update/behave_imp_weight` | π_proximal / π_behave | 异步训练中分布不匹配的离策略校正      |

**排除 `importance_weight` 偏差的故障：**

- 如果 `importance_weight/avg` 明显偏离 1，请减少 `ppo_n_minibatches`。
- 使用 `ppo_n_minibatches == 1` 时，理论上 `importance_weight` 应该恰好等于 1。
- 如果在 `ppo_n_minibatches == 1` 时偏差仍然存在（MoE 训练中常见），请在配置中添加
  `actor.megatron.use_deterministic_algorithms=1`。

**排除 `behave_imp_weight` 偏差的故障：**

- 确保设置了 `behav_imp_weight_cap`（推荐值：5）。
- 如果偏差仍然存在，请减少 `max_head_offpolicyness` 以减少样本过期程度。

### 序列长度指标

使用长轨迹训练时，监控这些指标以检测截断问题：

| 指标                     | 描述                                  |
| ------------------------ | ------------------------------------- |
| `ppo_actor/no_eos_ratio` | 在生成 EOS token 之前被截断的轨迹比例 |
| `ppo_actor/seq_len`      | 训练期间的平均序列长度                |

**排除高 `no_eos_ratio` 的故障：**

如果 `no_eos_ratio` 超过 0.05（5% 的轨迹被截断）：

- 增加 `max_new_tokens` 以允许更长的生成
- 使用动态过滤排除过长的轨迹

**监控序列长度增长：**

如果 `seq_len` 在训练过程中稳步增加，请密切注意 `no_eos_ratio`——增长的序列长度通常会导致更多截断。
