# Megatron Muon 与 AdamW 的 GSM8K GRPO 完整三轮对比

## 结论

本次在当前 AReaL 仓库的 `examples/math/gsm8k_rl.py` 上，直接使用未修改的
`examples/math/gsm8k_grpo_megatron.yaml`，让 AdamW 和 Muon 分别完成完整 GSM8K 训练集的 3 个 epoch。两组各
**29 step/epoch、87 次成功优化器更新**，退出码均为 0。学习率均保持基线的 `3e-6`。此前 `BASELINE_COMPARISON.md` 的
`train[:256]` 短测只有 3 次更新，不能代表这次完整训练。

最终模型在完整测试集上的评估 reward：AdamW 为 **4108/5276 = 0.77862**，Muon 为 **4065/5276 =
0.77047**。两者相差 43 个正确答案，即 0.815 个百分点。三轮评估都是同一批 1,319 道测试题，每题独立采样 4
次。两组均值在三轮内总体上升，但只有三个评估点，不能据此认定已经收敛或趋于平缓，也不能凭一次随机运行确定优化器效果优劣。

## Reward 细节

| Epoch | 训练 step | AdamW 训练正确数/29,696 | Muon 训练正确数/29,696 | AdamW 评估正确数/5,276 | Muon 评估正确数/5,276 |
| ----- | --------- | ----------------------- | ---------------------- | ---------------------- | --------------------- |
| 1     | 1–29      | 24,153（0.81334）       | 24,282（0.81769）      | 4,002（0.75853）       | 4,026（0.76308）      |
| 2     | 30–58     | 25,909（0.87247）       | 25,810（0.86914）      | 4,078（0.77293）       | 4,040（0.76573）      |
| 3     | 59–87     | 26,480（0.89170）       | 25,570（0.86106）      | 4,108（0.77862）       | 4,065（0.77047）      |

训练 reward 是参与更新的 rollout 原始 0/1 正确性奖励，不是更新后模型在测试集上的成绩。每轮使用 29 × 256 = 7,424 道训练题，每题生成 4
个答案，所以分母是 29,696；本地 GSM8K 训练集共有 7,473 道题，基线的 `drop_last=true` 每轮丢掉最后不足一个 batch 的 49
道题。评估 reward 是该轮更新后模型在 1,319 道固定测试题上的 5,276 次生成的正确率。两组运行分别采样，虽使用相同
`seed: 1`，也不能把随机生成视作逐答案配对实验。

只看均值会掩盖题目层面的变化。下表列出每轮 1,319 道题中，4 次生成分别答对 0、1、2、3、4 次的题数：

| Epoch | 优化器 | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
| ----- | ------ | --: | --: | --: | --: | --: |
| 1     | AdamW  | 157 |  98 | 103 | 146 | 815 |
| 1     | Muon   | 150 | 104 |  92 | 154 | 819 |
| 2     | AdamW  | 152 |  86 |  93 | 146 | 842 |
| 2     | Muon   | 162 |  91 |  90 | 135 | 841 |
| 3     | AdamW  | 163 |  78 |  84 | 114 | 880 |
| 3     | Muon   | 160 |  93 |  84 | 124 | 858 |

最终一轮 Muon 与 AdamW 对比，907 道题的四次正确数相同；Muon 在 197 道题上更多，在 215 道题上更少。按题配对估算的均值差为 -0.815
个百分点，近似 95% 区间为 \[-2.334，+0.704\] 个百分点；这个区间只反映本次测试题之间的差异，没有覆盖重复训练与生成随机性的全部波动。
因此这次结果支持“功能跑通、短期 reward 同一量级”，不支持“Muon 优于 AdamW”的结论。

## 训练速度与显存

| 指标                                        |       AdamW |        Muon |
| ------------------------------------------- | ----------: | ----------: |
| 87 步平均 `timeperf/train_step`             |    8.997 秒 |    9.436 秒 |
| 包括启动、rollout、评估、保存与退出的总耗时 |    1,990 秒 |    1,943 秒 |
| Actor GPU 每秒采样的最大值                  |  97,950 MiB |  91,124 MiB |
| Actor 四个 rank 的逐 rank 时间中位数的平均  |  70,604 MiB |  78,711 MiB |
| Rollout GPU 每秒采样的最大值                | 134,723 MiB | 130,165 MiB |

Muon 的平均训练步耗时多 0.438 秒，约慢 4.9%；两次端到端总时长受 rollout 与评估波动影响，不能当作纯优化器性能。 显存通过 `nvidia-smi`
每秒采样一次。AdamW 的单点峰值高于 Muon，但 AdamW 各 actor rank 的常态中位数较低；Muon 常态占用平均高约 8,107 MiB。普通 Muon
在 DP rank 间复制优化器状态，而基线 AdamW 使用分片状态，这是更高常态占用的结构性原因；序列长度、缓存与采样时刻也影响显存，不能把差值全部精确归因于优化器。

## 配置、环境与复现

两组都从同一 YAML 启动，只覆盖实验名、输出路径、本地模型与数据路径、现有 Slurm 分配中的
`scheduler.type=local`、`total_train_epochs=3`，并为 rollout 设置临时管理密钥。Muon 另外覆盖
`actor.optimizer.type=muon` 与
`actor.megatron.ddp.use_distributed_optimizer=false`。**没有覆盖**训练/验证集 split、batch
size、学习率、生成参数或 GRPO 参数。基线 actor 为 DP4/TP1，使用 GPU 0–3；SGLang rollout 为 DP4/TP1，使用 GPU
4–7。

模型：`/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct`；数据：
`/storage/openpsi/data/gsm8k`。实际环境：Slurm 作业 `973447`，单节点 `slurmd-5`、8 张 NVIDIA
L20X；Apptainer 镜像 `/storage/openpsi/images/areal-dev-sglang-0821.sif`；Python
3.12.3、PyTorch 2.9.1+cu129、Megatron-Core 0.19.0、Emerging Optimizers 0.3.0、SGLang
0.5.10.post1。

在当前仓库根目录和有效的 8 GPU Slurm 分配内复现，替换作业号与输出目录：

```bash
bash research/muon/run_full_3epoch.sh \
  "$PWD" "$PWD/research/muon/evidence/full-dataset-3epoch-20261003" 973447
python3 research/muon/summarize_full.py \
  "$PWD/research/muon/evidence/full-dataset-3epoch-20261003"
python3 research/muon/analyze_full_eval.py \
  "$PWD/research/muon/evidence/full-dataset-3epoch-20261003"
```

每组的精确命令、环境、UTC 起止时间、退出码、训练日志和每秒 GPU 显存采样在
`evidence/full-dataset-3epoch-20261003/{adamw,muon}/`。`comparison.json` 汇总所有 87 步的成功状态、
有限梯度范数、学习率和逐轮指标；`eval_detail.json` 从原始 JSONL 重新统计逐题分布并确认两组、三轮均为相同 1,319 道题。完整训练日志
`grpo.log` 在本机保留，对应的 `grpo.log.gz` 加入 Git 便于复查。原始 rollout 与 checkpoint
文件体积大，保留在本机上述输出目录，未加入 Git。

## 功能验收范围

实现选择 Megatron-Core 普通 `TensorParallelMuon`，适用的二维矩阵走 Muon；embedding、bias、norm 等由
`FusedAdam` 后备处理。DP 梯度由 Megatron DDP all-reduce 同步，Muon 状态在 DP rank 间复制， 并非 layer-wise
distributed Muon。已有单测检查参数映射、Adam/SGD 回归和不兼容组合的启动前错误； 多 GPU 测试在 DP2/TP1、DP2/TP2
上验证三步权重更新、梯度同步、模型/优化器/调度器 checkpoint 恢复并继续一步。完整 GSM8K 运行使用 BF16
DP4/TP1。详见短测报告中的测试命令与日志；此次补跑 针对训练轮次和 reward 有效性，没有改动优化器实现。
