# 检查点

本文档描述 AReaL 的检查点系统，用于在分布式 RL 训练期间保存模型以进行评估和容错恢复。

## 概述

AReaL 提供两种互补的检查点机制：

| 机制               | 用途                   | 格式                | 包含优化器/数据加载器状态 |
| ------------------ | ---------------------- | ------------------- | ------------------------- |
| **Saver**          | 导出模型用于评估或发布 | HuggingFace         | 否                        |
| **RecoverHandler** | 故障后恢复训练         | DCP（分布式检查点） | 是                        |

两种机制在训练期间自动调用，可分别通过 `config.saver` 和 `config.recover` 进行配置。

## 检查点格式

### HuggingFace 格式

由 `Saver` 用于模型导出：

- 标准 HuggingFace 模型格式（safetensors + config.json）
- 兼容 `transformers.AutoModel.from_pretrained()`
- 可上传到 HuggingFace Hub
- 不包含优化器状态

### DCP 格式（分布式检查点）

由 `RecoverHandler` 用于容错：

- 后端原生分布式检查点格式（`torch.distributed.checkpoint` 或 Megatron 分布式检查点）
- 跨所有 rank 分片以实现高效的并行 I/O
- 包含模型权重、优化器状态、RNG 状态等
- 后端特定：检查点仅与相同并行配置兼容
- 覆盖之前的检查点以节省磁盘空间

## 架构

```
PPOTrainer.train()
│
├── 训练循环
│   ├── Rollout、计算值、PPO 更新...
│   │
│   ├── _save_hf()                          # HuggingFace 导出
│   │   └── Saver.save()
│   │       └── engine.save(weight_format="hf")
│   │
│   └── _save_recover_checkpoint()          # 容错
│       └── RecoverHandler.dump()
│           └── engine.save(weight_format="dcp", with_optim=True)
│
└── 重启时
    └── RecoverHandler.load()
        ├── 恢复 dataloader、saver、evaluator 状态
        └── engine.load(weight_format="dcp", with_optim=True)
```

## Saver：HuggingFace 模型导出

[`Saver`](https://github.com/areal-project/AReaL/blob/main/areal/utils/saver.py) 定期以
HuggingFace 格式导出模型权重，用于评估或部署。

### 保存模式

`mode` 参数控制检查点的写入方式：

| 模式    | 行为                                                                                                                                                                                       |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `auto`  | 对 Archon 引擎使用异步，其他使用同步（默认）。对所有引擎都是零配置最优选择。                                                                                                               |
| `sync`  | 始终同步 `dcp.save()`。                                                                                                                                                                    |
| `async` | 始终使用进程内异步，带有固定内存暂存。仅适用于 Archon 引擎；其他引擎回退到同步并发出警告。额外的 CPU 固定内存与每 rank 模型分片大小成正比（例如，70B 模型在 8 GPU 上每 rank 约 17.5 GB）。 |

使用默认的 `auto` 模式，Archon 引擎用户自动获得异步检查点保存 —— 训练循环仅在检查点暂存到固定 CPU 内存时阻塞，实际磁盘 I/O 在后台进程中发生。

### 配置

通过 `config.saver` 配置：

| 参数          | 类型        | 默认值   | 描述                                |
| ------------- | ----------- | -------- | ----------------------------------- |
| `mode`        | str         | `"auto"` | 保存模式（见上文）。                |
| `freq_epochs` | int \| None | None     | 每 N 个 epoch 保存一次。None 禁用。 |
| `freq_steps`  | int \| None | None     | 每 N 步保存一次。None 禁用。        |
| `freq_secs`   | int \| None | None     | 每 N 秒保存一次。None 禁用。        |

配置示例：

```yaml
saver:
  freq_epochs: 1      # 每个 epoch 结束时保存
  freq_steps: null    # 禁用
  freq_secs: null      # 禁用
  # mode 默认为 "auto" - Archon 用户自动获得异步
```

当满足 epoch/步数/时间任一条件时触发保存。

### 输出位置

检查点保存到：

```
{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/default/
└── epoch{E}epochstep{S}globalstep{G}/
    ├── config.json
    ├── model.safetensors（或 model-00001-of-00002.safetensors 等）
    ├── tokenizer.json
    └── ...
```

### 使用方式

使用标准 HuggingFace API 加载保存的检查点：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint/epoch0epochstep99globalstep99"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/checkpoint/epoch0epochstep99globalstep99"
)
```

## RecoverHandler：容错

[`RecoverHandler`](https://github.com/areal-project/AReaL/blob/main/areal/utils/recover.py)
通过保存完整训练状态实现故障后恢复训练。

### 配置

通过 `config.recover` 配置：

| 参数          | 类型        | 默认值     | 描述                                      |
| ------------- | ----------- | ---------- | ----------------------------------------- |
| `mode`        | str         | "disabled" | 恢复模式："on"/"auto" 或 "off"/"disabled" |
| `freq_epochs` | int \| None | None       | 每 N 个 epoch 检查点                      |
| `freq_steps`  | int \| None | None       | 每 N 步检查点                             |
| `freq_secs`   | int \| None | None       | 每 N 秒检查点                             |
| `retries`     | int         | 3          | 启用恢复时的恢复重试次数                  |

#### 恢复模式

| 模式                | 行为                         |
| ------------------- | ---------------------------- |
| `on` 或 `auto`      | 如果存在有效检查点则自动恢复 |
| `off` 或 `disabled` | 不进行检检查点或恢复         |

启用恢复（`on`/`auto`）时，系统将：

1. 定期保存恢复检查点（模型权重、优化器状态、dataloader 位置）
1. 重启时自动从最后一个有效检查点恢复
1. 失败时重试最多 `retries` 次

配置示例：

```yaml
recover:
  mode: on            # 或 "auto"（向后兼容）
  freq_steps: 100     # 每 100 步检查点
  retries: 3
```

### 保存的内容

RecoverHandler 保存完整训练状态：

| 组件            | 内容                                  |
| --------------- | ------------------------------------- |
| 模型权重        | DCP 格式，跨 rank 分片                |
| 优化器状态      | 动量、方差（Adam）、学习率调度器      |
| RNG 状态        | Python、NumPy、PyTorch、CUDA 随机状态 |
| Dataloader 状态 | 数据集中的当前位置                    |
| 训练进度        | Epoch、step、global_step 计数器       |
| 辅助状态        | Saver、Evaluator、StatsLogger 状态    |

### 输出位置

恢复检查点保存到：

```
{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/
├── default/
│   └── recover_checkpoint/     # 模型 + 优化器（DCP 格式）
│       ├── __0_0.distcp
│       ├── __1_0.distcp
│       └── ...
├── critic/                     # 如果使用 critic
│   └── recover_checkpoint/
└── recover_info/               # 元数据
    ├── step_info.json
    ├── saver_info.json
    ├── evaluator_info.json
    ├── stats_logger_info.json
    ├── checkpoint_info.json
    └── dataloader_info.pkl
```

### 恢复过程

恢复训练时：

1. `RecoverHandler.load()` 恢复所有保存的状态（如有）
1. 训练从 `last_step_info.next().global_step` 继续
1. 推理引擎权重同步以匹配恢复的状态

## 最佳实践

### 频率指南

| 场景           | 推荐设置                                  |
| -------------- | ----------------------------------------- |
| 长时间训练     | `freq_epochs: 1` 或 `freq_steps: 1000`    |
| 不可预测的时间 | `freq_secs: 7200`                         |
| 不稳定的集群   | `freq_steps: 100` 配合 `recover.mode: on` |
| 磁盘空间有限   | 较低频率，依赖最终检查点                  |
| 调试           | `freq_steps: 1` 以快速迭代                |

### 磁盘空间注意事项

- **Saver**：每次保存创建新目录。高频率会消耗大量空间。
- **RecoverHandler**：覆盖之前的检查点。同时只存在一份副本。

### 恢复技巧

1. **验证检查点有效性**：检查 `recover_info/step_info.json` 获取最后保存的步数
1. **需要相同配置**：DCP 检查点需要相同的并行配置、实验名称和试次名称
1. **干净重启**：删除 `recover_info/` 目录以全新开始
