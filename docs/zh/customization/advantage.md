# 自定义优势计算

`actor.advantage` 可以选择完整优势计算函数，也可以单独替换计算阶段。默认配置保留现有 GAE、奖励归一化和 PRM 行为。配置只保存导入路径和可 JSON
序列化的参数，各 actor worker 在初始化时解析函数。

```yaml
actor:
  advantage:
    compute:
      implementation: default
      kwargs: {}
    outcome_reward_shaping: null
    estimator: null
    process_adv_shaping: null
```

默认流程依次执行长度惩罚、outcome shaping、奖励缩放与归一化、时序估计、优势归一化、process advantage shaping。Loss
aggregation 和 optimizer minibatch 调度保持不变。

| 阶段                     | 内置实现                               | 默认行为                         |
| ------------------------ | -------------------------------------- | -------------------------------- |
| `outcome_reward_shaping` | `identity`                             | 保留 outcome reward              |
| `estimator`              | `token_gae`、`turn_gae`                | 跟随 `actor.gae_timestep_unit`   |
| `process_adv_shaping`    | `additive`、`gvpo`、`process_weighted` | 使用现有 PRM 选择，否则 additive |

每个阶段也支持可导入的 Python 函数路径和 `kwargs`。`auto` 使用阶段默认实现；process 的 `auto` 不接受 kwargs。显式 process
选择优先于调用时传入的 PRM 选择；显式 GVPO 参数覆盖旧参数，未指定的参数沿用旧值。现有 PRM 配置仍受支持。

```yaml
actor:
  token_rewards_as_adv: true
  advantage:
    estimator:
      implementation: turn_gae
      kwargs: {}
    process_adv_shaping:
      implementation: gvpo
      kwargs:
        negative_scale: 0.2
        zero_penalty: 0.4
```

Turn GAE 要求 rollout 提供 `turn_ids`。其 token KL 惩罚只影响 actor advantages，不进入 critic
returns；token GAE 则将 KL 纳入递推。自定义 GAE lambda 接收实际 estimator 对应的时间单位长度，即使 estimator 覆盖了
`gae_timestep_unit`。原有 `GAELambdaContext` 函数仍可用，`GAELambdaInputs` 是同一类型的别名。

## 阶段函数

Outcome 函数接收 `areal.trainer.ppo.advantage.OutcomeRewardInputs`。其中 `data` 是长度惩罚后的 rollout
数据，`config` 为 actor 配置，`meta` 包含轨迹与 logical rollout 分组信息。返回
`OutcomeResult(rewards=...)`，每个 tensor row 对应一个浮点奖励：

```python
from areal.trainer.ppo.advantage import OutcomeResult, OutcomeRewardInputs


def scale_outcome(inputs: OutcomeRewardInputs, *, factor: float) -> OutcomeResult:
    return OutcomeResult(rewards=inputs.data["rewards"] * factor)
```

配置 `implementation: my_package.my_module.scale_outcome` 和
`kwargs: {factor: 0.5}`。该模块需要在所有 actor worker 上可导入。函数必须保持行顺序、分组和设备，不应原地修改输入 mapping 或
tensor。

奖励归一化使用显式 logical rollout 参考奖励时，自定义 shaping 还必须返回
`OutcomeResult.normalization_rewards`：每个 logical rollout 一个变换后的参考值，尚未经过 bias、scaling 和
clipping。原始参考值与行映射在 `meta.rollout_groups` 中。没有显式参考值时，同一 logical rollout
内各行奖励相同即可推导参考值；各行不同则需要提供显式 shaped references。`meta.traj_group_sizes` 统计每个 prompt 的
tensor row 数，`meta.logical_group_sizes` 统计 logical rollout 数。Shaped references 要求提供
`concat_batch` 生成的完整 metadata，不能改变行数或分组成员。分布式插件应使用框架提供的 metadata，不应自行构造不完整的分组信息。

时序函数接收 `areal.trainer.ppo.advantage_algorithms.TemporalInputs`，返回
`TemporalResult(advantages, returns, total_rewards)`。输入已按 next-token prediction
对齐，outcome、KL 和折叠后的 process rewards 分开传递。显式 `bootstrap_values` 来自每条轨迹实际最后一个有效 token。自定义
estimator 可调用 `areal.trainer.ppo.gae.compute_gae_lambda`，传入自身的时间单位以解析 callable
lambda。`total_rewards` 是按 token 对齐的监控奖励。

Process 函数接收 `ProcessAdvantageInputs`，返回 advantage tensor。输入包含归一化后的 advantages、对齐后的直接
token rewards（可能为 `None`）和 loss mask，不修改 critic returns。内置 GVPO 与 process-weighted 要求
`token_rewards_as_adv=True`；process-weighted 还要求 `mask_no_eos_with_zero=False`。

## 替换完整计算

将 `compute.implementation` 设为接收 `AdvantageInputs`、返回 `AdvantageResult`
的函数路径。完整替换不能与阶段配置同时使用。默认 compute 不接受 kwargs，应通过阶段参数配置。

`AdvantageResult` 包含 advantages、returns、loss mask、对齐后的 log probabilities、outcome
rewards、total rewards 和 KL rewards。输出必须具有正确的形状和设备，使用有限浮点数；mask 必须为二值。Identity outcome
阶段也保留旧工作流中整数任务奖励的类型。可选 `extras` 仅允许包含
`input_token_loss_mask`、`is_truncated`、`mopd_behavior_logprobs`、`prox_logp` 和
`ref_logp`，不能覆盖 input IDs、values、original rewards 等 rollout
字段。完整替换函数负责对齐与训练模式需要的派生信息，包括与 MOPD 组合时保存不可变的 behavior log probabilities。
