# 直接偏好优化（DPO）

最后更新：2026 年 4 月 22 日

## 概述

**直接偏好优化（Direct Preference Optimization, DPO）** 是一种离线对齐算法，直接在人类偏好数据（chosen / rejected 对）上优化语言模型，无需奖励模型，也无需在线 RL 采样。

与 RLHF（PPO）相比，DPO 更简单（无奖励模型、无价值网络、无在线生成）、更稳定（单一监督式损失）、更高效（每个 batch 仅两次前向 + 一次反向，即策略 + 参考）。AReaL 实现了基于 FSDP2 的 DPO，支持参考模型共卡部署。

**论文**：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)（Rafailov 等，NeurIPS 2023）

## 核心思想

### DPO 目标函数

给定偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$，其中 $y_w$ 为 chosen 回复、$y_l$ 为 rejected 回复，DPO 优化：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) =
-\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[\log \sigma\!\left(\beta \left(
\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
- \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}
\right)\right)\right]
$$

$\pi_\theta$ 为训练中的策略模型，$\pi_{\text{ref}}$ 为冻结的参考模型，$\beta$ 控制 KL 惩罚强度。

该目标可通过将 KL 正则化 RLHF 最优策略的闭式解代入 Bradley-Terry 偏好模型推导得出——奖励函数由策略与参考隐式定义，因此无需单独训练奖励模型。

AReaL 通过 `loss_type` 支持两种损失：默认的 sigmoid 形式和 **IPO**（Azar 等 2023），后者以逐 token 平均的平方损失逼近固定边际 $\frac{1}{2\beta}$。IPO 变体在计算平方损失前先按 completion 长度归一化 logratio（逐 token 平均），与 TRL 的经作者确认的实现一致。

### 隐式奖励

训练过程中以 $r(x, y) = \beta (\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x))$ 作为隐式奖励。**奖励边际** $r(x, y_w) - r(x, y_l) > 0$ 表示模型正确偏好 chosen 回复，**奖励准确率**即边际为正的样本对比例。

## 运行示例

### 单机训练（HH-RLHF）

```bash
python3 examples/alignment/hhrlhf_dpo.py \
  --config examples/alignment/hhrlhf_dpo.yaml \
  scheduler.type=local
```

配置文件 `examples/alignment/hhrlhf_dpo.yaml` 的关键片段：

```yaml
actor:
  backend: "fsdp:d8p1t1"
  path: Qwen/Qwen2.5-7B            # 遵循原论文：在 base 模型上训练
  beta: 0.1                        # KL 惩罚系数
  dtype: bfloat16
  disable_dropout: true            # DPO 稳定性所必需
  mb_spec:
    granularity: 2                 # DPO 必须为 2：chosen + rejected 成对调度
  optimizer:
    lr: 5e-6
    lr_scheduler_type: cosine
    warmup_steps_proportion: 0.1

ref:
  backend: ${actor.backend}
  path: ${actor.path}
  optimizer: null                  # 冻结模型
  scheduling_strategy:
    type: colocation
    target: actor                  # 与 actor 共卡以节省显存

train_dataset:
  batch_size: 8
  path: Anthropic/hh-rlhf
  type: dpo
  max_length: 2048
```

`get_hhrlhf_dpo_dataset`（`areal/dataset/hhrlhf.py`）直接对 chosen/rejected 原始文本分词，并以 token 级最长公共前缀作为 prompt 边界。HH-RLHF 数据对共享相同的多轮 prompt，仅最后一条 assistant 回复不同，公共前缀恰为 prompt。

### 多机训练（Ray）

```bash
python3 examples/alignment/hhrlhf_dpo.py \
  --config examples/alignment/hhrlhf_dpo.yaml \
  cluster.n_nodes=2 cluster.n_gpus_per_node=8 \
  cluster.fileroot=/path/to/nfs \
  scheduler.type=ray
```

## 关键参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `actor.beta` | `0.1` | KL 惩罚系数。越大越接近参考模型。典型范围 0.05–0.5。 |
| `actor.loss_type` | `"sigmoid"` | 损失变体。`"sigmoid"` 为原始 DPO；`"ipo"` 使用逐 token 平均的平方损失（Azar 等 2023）。 |
| `actor.optimizer.lr` | `5e-6` | 学习率。DPO 对 LR 敏感，建议 5e-7 – 5e-6。 |
| `actor.disable_dropout` | `true` | 禁用 dropout 以确保 log 概率计算确定性。 |
| `actor.mb_spec.granularity` | `2` | 微批粒度。DPO 必须为 2（chosen+rejected 成对）。 |
| `ref` | — | 参考模型配置（必填）。 |

训练过程中会记录 `dpo/loss`、`dpo/chosen_reward`、`dpo/rejected_reward`、`dpo/reward_accuracy`、`dpo/reward_margin` 等指标（前缀 `dpo/`）。

## 参考

- Rafailov 等（2023）. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Azar 等（2023）. *A General Theoretical Paradigm to Understand Learning from Human Feedback*. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- [Anthropic HH-RLHF 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)
