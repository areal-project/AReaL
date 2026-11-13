# 单样本异步优化 (SAO)

最后更新：2026年11月11日

SAO（[arXiv:2607.07508](https://arxiv.org/abs/2607.07508)；*Single-Rollout
Asynchronous Optimization for Agentic Reinforcement Learning*）是在 **现成 AReaL
PPO** 上的配方：每个 prompt 只采 1 条轨迹（G=1）、启用 value critic，以及按重要性
比值带做 token 级 rejection sampling（DIS 风格 mask）。

## 和 GRPO 的对比

| 框架变量 | 典型 GRPO 示例 | SAO 示例 |
|------|----------------|----------|
| `gconfig.n_samples` | ≥4 / 8 | **1** |
| Critic | 关闭 | **开启**（与 actor 同卡 value critic） |
| Advantage 基线 | group reward norm | batch `adv_norm`（无 group） |
| `actor.rejection_sampling` | 常只设 upper / 更松 | **mask** 带 `[0.3, 5.0]` |

本示例**只**用现成 AReaL API（critic 块 + `rejection_sampling`），不改引擎代码。

**不在本 PR：** 论文的每步多次 critic 更新（K=2）与 skip-observation GAE。现成
critic 是单次更新。

## 如何复现

与其他 math 示例共用入口：

| Backend | CMD |
|---------|-----|
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=local` |
| **ray** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=ray` |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=slurm` |

配置：[`examples/math/gsm8k_sao.yaml`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_sao.yaml)。

## 报告数字

社区 MATH-hard 在线实验（非 AReaL CI、非论文表内绝对分）：

- 模型：`Qwen3-4B-Instruct-2507`
- 数据：MATH train（约 7.5k）转成 GSM8K 风格 schema
- 规模：三机共 24 GPU（每机八卡），`total_train_steps=1000`
- 指标：训练流上的在线 `acc = correct / (correct + incorrect)`
- **当时未做：** held-out MATH-500 / AIME

| 设定 | G | Critic | DIS 带 | Steps | 在线 acc（开局 → 末窗） |
|------|---|--------|--------|-------|-------------------------|
| SAO（报告） | 1 | 有 (K=2)† | mask [0.3, 5] | 1000 | ~40% → **~49%**（951–1000） |
| GRPO | 8 | 无 | 无 | 268‡ | ~39% → **~15%** |
| GRPO+DIS | 8 | 无 | 有 | 1000 | ~40% → **~57%** |

† 报告用的是 K=2；本仓库 yaml 在合入独立 PR 前是现成单次 critic。
‡ 裸 GRPO 崩塌后提前停。G=1 与 G=8 **不能**直接比绝对分（每步轨迹数差 8 倍）。在同
一 G=1 预算下，该报告里 SAO 高于「仅 DIS」(~43%) 与裸 running-mean (~10%)。

仓库里的 `gsm8k_sao.yaml` 沿用共享的 **1.5B / 8 卡 / GSM8K** 模板，方便冒烟。要对齐
上表，需要把模型、数据与集群扩到上述 MATH-hard 设定。

## 参考

- SAO 论文：[arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- AReaL：[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)
