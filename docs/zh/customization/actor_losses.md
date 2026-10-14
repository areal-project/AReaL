# 可组合的 actor loss

`actor.losses` 配置具有独立 support、归一化方式和系数的命名 token 目标。 空映射保留现有 PPO/PRM/MOPD
目标；非空映射替换隐式目标，需要策略优化时必须显式包含 policy 项。

```yaml
actor:
  loss_aggregation: token_mean
  losses:
    policy:
      implementation: policy
      reduction: prompt_mean
    exploration:
      implementation: entropy
      coefficient: 0.01
  megatron:
    entropy_requires_grad: true
```

内置实现包括 `policy`（沿用 actor 的 PPO、SAPO 或 CISPO 设置）、`entropy`（负熵）和 `mopd`（多教师蒸馏）。entropy
必须可求梯度；Megatron 需要启用 `entropy_requires_grad` 并使用兼容的 LM-head 配置。插件在每个 actor worker 上解析一次。

## 配置与优先级

每项支持 `implementation`、`enabled`、`coefficient`、`reduction`、`mask_key` 和
`kwargs`。系数必须有限且非负。禁用或零系数项不导入、不执行；至少需要一个有效项。 kwargs 必须可 JSON 序列化，不得覆盖运行时参数。

未指定 `reduction` 时继承 `actor.loss_aggregation`，这是唯一的默认入口。 源分支的 `default_loss_reduction`
应迁移为 `loss_aggregation`。 显式内置 kwargs 覆盖 actor 默认值，但影响预处理的 `prox_logp_method` 与
`use_decoupled_loss` 必须与 actor 一致。

`mask_key` 默认是 `loss_mask`，指定与其同形状的二值 token mask。两者的交集在 optimizer 拆分前确定 support
和归一化分母；后续 rejection filtering 仅改变分子。零奖励不代表缺少 support。自定义 mask 必须已经对齐预测位置（next-token
label）， 与 advantage 计算后的 `loss_mask` 一致，不会自动移位；原始 rollout `turn_ids` 会单独移位。

## 归一化方式

| reduction     | 目标                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| `token_mean`  | 有效 token 的平均值。                                                      |
| `seq_mean`    | 先计算每条有效 response 的 token 均值，再对 response 求均值。              |
| `constant`    | token 总和除以有效 response 数与 `actor.loss_aggregation_divisor` 的乘积。 |
| `turn_mean`   | 先计算每个有效 turn 的 token 均值，再对 turn 求均值。                      |
| `prompt_mean` | 有效 prompt 等权，各 prompt 内有效 token 等权。                            |

每项使用自己的 support 与分母。全局无有效 token 的命名项贡献可微分的零。 `turn_mean` 需要原始 token 对齐的 `turn_ids`，actor
将它移位到预测位置。 turn 权重和数量在 packing 或 microbatch 拆分前准备。 不同 response 可以复用 ID，但一条 response 的有效
token ID 必须非负、单调不减且小于 padded 序列长度。

`prompt_mean` 保留 main 按 response 划分 optimizer step 的调度，同一 prompt 可以跨 step。 设实际 step 数为
K，该项在整个 update 的全局有效 prompt 数为 G，每步使用完整 update 的 权重并乘以
K/G。固定参数时，各步贡献的平均值恢复完整目标；这不意味着多次参数更新与一次大步更新相同。 全空 group 不计入 G。

引擎接收每个名称独立的分子和 normalizer。系数仅乘入分子一次。 所有 rank、microbatch、空项和 transport-only rank
均保留同一名称集合。

## 自定义 token loss

实现可导入的 dotted callable，并显式声明 keyword-only 运行时参数：

```python
import torch


def custom_loss(*, outputs, batch, mask, scale=1.0):
    return torch.where(mask, -scale * outputs["entropy"], 0.0)
```

outputs 包含 `logprobs`、`entropy` 和可用的 vocabulary 统计，batch 包含准备好的 advantage 和当前策略版本。两个
mapping 均只读，也不要原地修改其中的 tensor。 返回与 mask 同形状、同设备、可微分的浮点 tensor，有效位置必须有限。
不要自行归约，也不要在过滤分子后重新计算分母。回调必须兼容 padded 和 packed 布局。

## 兼容性与限制

- 未启用命名 loss 时，原目标与诊断行为保持不变。
- 每个命名项记录独立的加权 loss 均值；policy 还记录 clipping、importance weight、KL 和 rejection
  指标，但不会为每个项完整复制原有 proximal approximation 诊断集合。
- 命名 MOPD 使用不可变的 `mopd_behavior_logprobs`，不使用重算后的 PPO logprobs。 命名项的 coefficient 和
  `importance_ratio_cap` kwarg 决定实际目标。 `mopd.loss` 仍决定是否准备教师目标和 RL advantage，但不额外叠加隐式
  loss，也不覆盖命名系数。 使用命名 MOPD 必须启用教师目标；policy 项不能配合纯蒸馏预处理。
- 显式命名 loss 不支持旧式 `teacher_logp` 和依赖 forward 的 M2PO 选择；原有未命名路径继续支持。
- tree-packed 训练仅支持 `token_mean`。
- 独立 process reward loss 路由、efficiency presets 和 turn reward 数据接入不属于本接口。 现有 PRM
  advantage shaping 继续通过 advantage pipeline 使用。
