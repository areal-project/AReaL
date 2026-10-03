# Megatron Muon 的 GSM8K GRPO 三步短测报告

此文件记录每轮仅一个 batch、总共三个优化器 step 的短程调试。完整训练集的三轮复跑每轮有 29 个 step、共 87 个 step；其正式结果见
`FULL_DATASET_COMPARISON.md`。

## 结论与范围

验收代码位于 `/storage/openpsi/users/fenghui/projects/AReaL` 的
`feature/megatron-muon-validated` 分支。实现选择 Megatron-Core 原生普通
`TensorParallelMuon`：适用的二维矩阵使用 Muon，embedding、bias、norm 等使用 Adam 标量后备优化器。DP 梯度由 Megatron
DDP all-reduce 同步，Muon 优化器状态在 DP rank 间复制；此实现没有启用 layer-wise distributed Muon。

在 Slurm 作业 `973447` 的单节点 8 张 NVIDIA L20X 上，Muon 和 AdamW 均完成 3 个 GSM8K GRPO epoch、3
次非零梯度的成功更新、rollout、奖励和逐 epoch 评估， 退出码均为 0。下表来自本目录
`evidence/baseline-aligned-3epoch-20261003/comparison.json`， 而不是先前的独立检出目录。

| 指标                             |                       AdamW |                        Muon |
| -------------------------------- | --------------------------: | --------------------------: |
| 训练正确答案数，第 1/2/3 epoch   |             704 / 744 / 690 |             709 / 763 / 684 |
| 训练 task reward，第 1/2/3 epoch | 0.68750 / 0.72656 / 0.67383 | 0.69238 / 0.74512 / 0.66797 |
| 评估正确答案数，第 1/2/3 epoch   |                14 / 16 / 15 |                14 / 13 / 14 |
| 评估 reward，第 1/2/3 epoch      |    0.8750 / 1.0000 / 0.9375 |    0.8750 / 0.8125 / 0.8750 |
| 平均 `timeperf/train_step`       |                 9.405 秒/步 |                 9.475 秒/步 |
| 含启动、评估、保存、退出的总耗时 |                      320 秒 |                      312 秒 |
| Actor GPU 显存采样峰值           |                  77,984 MiB |                  79,926 MiB |
| Rollout GPU 显存采样峰值         |                 119,443 MiB |                 119,327 MiB |

每轮训练使用 256 道题、每题生成 4 个答案，共 1024 个答案。`ppo_actor/task_reward/avg` 是参与该次更新的原始 0/1
正确性奖励均值，不是 loss，也不是该次更新后模型的评估成绩。上表正确数来自日志的 `ppo_actor/correct_n_seqs`。两组训练奖励都在第 2 轮升高、第
3 轮回落；总共只有 3 次更新，**没有证据表明奖励趋于平缓**。

最终模型在第 3 轮更新后的评估 reward 为 AdamW **15/16 = 0.9375**、Muon **14/16 = 0.8750**，相差一个答案。原始评估
rollout 按题统计如下；每格是 4 次生成中的正确数：

| 评估轮次 | 日志题号 | AdamW 逐题正确数 | Muon 逐题正确数 |
| -------- | -------- | ---------------- | --------------- |
| 1        | 0–3      | 4, 4, 2, 4       | 4, 3, 4, 3      |
| 2        | 4–7      | 4, 4, 4, 4       | 4, 4, 2, 3      |
| 3        | 8–11     | 4, 4, 3, 4       | 4, 3, 3, 4      |

运行脚本为了缩短验收时间使用 `valid_dataset.split="test[:4]"`。三轮评估的是**同样 4
道题**：日志的题号逐轮递增，但核对题目内容及其哈希后确认 0/4/8、1/5/9、2/6/10、3/7/11 分别是同一道题。4 道题的评估和随机生成的 16
个答案仍不足以判断两个优化器的效果优劣或奖励是否趋于平缓；需要更大的固定验证集、更多更新步和重复运行。逐题数据来自两组
`results/logs/root/*/trial0/eval-rollout/{1,2,3}/*.jsonl` 的 `original_reward` 字段。

训练步耗时差约 0.7%，actor 显存采样峰值差 1,942 MiB（约 2.5%）。这一次两者 速度与显存接近。Muon 使用复制式状态，AdamW
基线使用分片式状态，所以 Muon 存在更高 actor 显存需求的结构性原因；1 秒一次的 `nvidia-smi` 采样还受到序列 长度和缓存峰值影响，不能把 1,942
MiB 全部精确归因于优化器状态。现有 reward 数据不能证明长期收敛或统计优势。

同一代码提交在独立检出目录还完成过一次相同基线覆盖项的复跑，原始摘要与日志已复制到
`evidence/baseline-aligned-3epoch-20261002/`。那次 AdamW/Muon 的平均训练步为 9.580/9.764 秒，actor
显存采样峰值为 73,160/86,602 MiB；训练 reward 走势仍接近。两次显存差值分别为 1,942 和 13,442
MiB，说明一秒采样的峰值随当次序列长度、PyTorch 缓存和执行过程波动明显。复制式 Muon 状态相对分片 AdamW
是持续存在的内存成本，但现有采样不能单独测出它占这两个差值中的多少。

## 配置、环境与复现

两组都直接读取未改动的 `examples/math/gsm8k_grpo_megatron.yaml`，由 `examples/math/gsm8k_rl.py`
运行。命令行共同设置本地模型/数据/输出路径、 `scheduler.type=local`、`total_train_epochs=3`、训练切片 `train[:256]`
和 验证切片 `test[:4]`，并为 rollout 生成临时管理密钥。基线 batch size 256、 每题 4 个采样、生成长度、8 GPU 布局、GRPO
参数和学习率 `3e-6` 均保持原值。 Muon 只另外设置 `actor.optimizer.type=muon` 与必需的
`actor.megatron.ddp.use_distributed_optimizer=false`。每个 epoch 恰有一个完整 batch 和一次训练更新。actor
为 DP4/TP1，使用 GPU 0–3；SGLang rollout 为 DP4/TP1，使用 GPU 4–7。

模型：`/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct`；数据：
`/storage/openpsi/data/gsm8k`。环境是 Python 3.12.3、PyTorch 2.9.1+cu129、 Megatron-Core
0.19.0、Emerging Optimizers 0.3.0、SGLang 0.5.10.post1； GPU 驱动 570.148.08。未改动 CUDA 或驱动。

在此分支根目录、持有有效 Slurm 作业时执行：

```bash
bash research/muon/run_baseline_aligned.sh \
  "$PWD" "$PWD/research/muon/evidence/baseline-aligned-3epoch-20261003" 973447
python3 research/muon/summarize_baseline.py \
  "$PWD/research/muon/evidence/baseline-aligned-3epoch-20261003"
```

每组的精确命令、环境、退出码、UTC 起止时间、训练日志和每秒 GPU 显存样本分别 保存在
`evidence/baseline-aligned-3epoch-20261003/adamw/` 与 `muon/`； `comparison.json`
给出未舍入值。汇总脚本断言三条 epoch 记录、全部成功更新、 有限标量、正的权重同步耗时、评估记录、正常结束及退出码 0。

## 优化器与恢复验证

`tests/test_megatron_optimizer_config.py` 覆盖 Muon 参数映射、Adam/SGD 回归；
`tests/test_megatron_muon_config.py` 覆盖 YAML 解析、参数合法性及启动前兼容性 报错。分布式测试
`tests/test_megatron_muon_distributed.py` 在 DP2/TP1 和 DP2/TP2 上检查矩阵/标量参数分组、DP rank 同步、有限
loss/梯度范数/学习率、 连续三步矩阵权重变化、Muon 动量状态、模型/优化器/调度器检查点，以及从新建
优化器恢复后与未中断运行完全一致的第四步。检查点管理器明确允许此复制式 优化器的 sharded state dict，并避免给它传入分片 Adam 专用元数据。

本次六个 rank 的 JSON 报告均标记 `state_restored=true` 和 `fourth_update_identical=true`。每个 rank
的分组包含 112 个 `TensorParallelMuon` 矩阵参数及 86 个 `FusedAdam` 后备参数；调度器步数由 3 恢复并继续到 4。

运行范围为单节点 BF16 与 TP 的 blockwise 模式。FP16、LoRA、分片优化器 开关、参数收集重叠、FP8、精度感知状态和不适配的 AWEX
路径会在初始化前 报错。FP32、其他 TP 模式、多节点和异步检查点尚无运行验收，不能据此宣称支持。

在上述容器与 Python 环境中执行的测试命令：

```bash
python -m pytest -q tests/test_megatron_muon_config.py \
  tests/test_megatron_optimizer_config.py tests/test_megatron_async_save.py
AREAL_MUON_TEST_MODEL=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct \
  python -m pytest -q -s tests/test_megatron_muon_distributed.py
pre-commit run --all-files
```

输出保存在同一验收目录的 `pytest.log`（66 passed、2 skipped）、`distributed.log`（2
passed）、`distributed-ranks/` 和 `pre-commit-final.log`。正式 GRPO 运行默认不保存恢复检查点；恢复能力由上述独立多
GPU 测试实际保存并恢复后继续一步验证。
