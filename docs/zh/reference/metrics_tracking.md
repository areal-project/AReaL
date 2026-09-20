# 指标跟踪

AReaL 提供统一的指标跟踪系统，处理分布式训练和 Rollout 工作器的统计信息收集。该系统支持针对其各自使用场景优化的两种不同范式：**流式指标**用于异步
Rollout 工作流，**批量指标**用于同步训练更新。

## 核心组件

指标系统围绕 `areal.utils.stats_tracker` 构建，提供：

- **命名跟踪器**：不同组件的隔离指标命名空间
- **层级作用域**：将指标组织成逻辑组
- **分布式聚合**：跨工作器自动归约
- **多种归约类型**：支持平均值、求和、最小/最大值和标量

```python
from areal.utils import stats_tracker

# 默认跟踪器（训练指标）
stats_tracker.scalar(learning_rate=0.001)

# 命名跟踪器（Rollout 指标）
stats_tracker.get("rollout").scalar(reward=0.5)
```

## 两种日志范式

### 流式指标（Rollout 工作器）

Rollout 工作器异步执行工作流，每个工作流独立记录指标。这种流式方法自然地处理可变完成时间。

**特性：**

- 每个工作流在完成时单独记录标量
- 指标在工作器进程内的列表中累积
- 记录期间工作器之间无需同步
- 归约在导出时通过控制器完成

**来自 `RLVRWorkflow` 的示例：**

```python
# areal/workflow/rlvr.py
async def _collect_samples(self, engine, req, prompt_str, task_data):
    resp = await engine.agenerate(req)
    reward = await self._compute_rewards(resp, prompt_str, task_data)

    # 记录单个标量 - 追加到内部列表
    # `workflow_context.stat_scope()` 自动区分评估/训练作用域
    stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

    return resp, reward
```

你可以在自定义工作流中记录任何其他标量，例如：

```python
async def run(self, data, **extra_kwargs):
    # `workflow_context.stat_scope()` 自动区分评估/训练作用域
    stats_tracker.get(workflow_context.stat_scope()).scalar(num_turns=num_turns, max_tokens=max_tokens, reward=reward)
    return reward
```

**控制器聚合：**

`RolloutController` 从所有工作器收集统计信息并计算加权平均值：

```python
# areal/infra/controller/rollout_controller.py
def export_stats(self) -> dict[str, float]:
    all_raw_stats = self._collective_rpc(method="export_stats")

    # 使用计数作为权重进行聚合
    stats, counts = defaultdict(float), defaultdict(int)
    for raw_stats in all_raw_stats:
        for k, v in raw_stats.items():
            if k.endswith("__count"):
                counts[k] += v
            else:
                stats[k] += v * raw_stats.get(k + "__count", 0)

    # 计算加权平均值
    return {k: v / counts[k + "__count"] for k, v in stats.items()
            if counts.get(k + "__count", 0) > 0}
```

### 批量指标（训练引擎）

训练引擎在数据并行 rank 之间同步处理批次。指标作为带布尔掩码的张量记录，在导出时跨所有 rank 归约。

**特性：**

- 记录带分母掩码的完整批次张量
- 支持每 token 和每序列统计
- All-reduce 同步确保各 rank 统计信息一致
- 多种归约类型：`AVG_MIN_MAX`、`AVG`、`SUM`、`MIN`、`MAX`

**来自 `PPOActor` 的示例：**

```python
# areal/trainer/ppo/actor.py
def ppo_update(self, data):
    loss_mask = data["loss_mask"].bool()
    reward_score = data["rewards"]

    # 定义分母（布尔掩码）
    stats_tracker.denominator(
        n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
        n_valid_tokens=loss_mask,
    )

    # 使用分母引用记录张量指标
    stats_tracker.stat(
        advantages=data["advantages"],      # [batch, seq_len]
        kl_rewards=data["kl_rewards"],      # [batch, seq_len]
        denominator="n_valid_tokens"
    )

    stats_tracker.stat(
        task_reward=reward_score.float(),   # [batch]
        seq_len=seqlens.float(),            # [batch]
        denominator="n_seqs"
    )
```

**导出行为：**

```python
# areal/engine/fsdp_engine.py
def export_stats(self) -> dict[str, float]:
    # 跨数据并行组 all-reduce
    return stats_tracker.export_all(reduce_group=self.data_parallel_group)
    # 所有 DP rank 接收相同的结果
```

## API 参考

### 记录方法

| 方法                          | 使用场景         | 示例                                     |
| ----------------------------- | ---------------- | ---------------------------------------- |
| `scalar(**kwargs)`            | 单个浮点值       | `scalar(lr=0.001, eps=0.2)`              |
| `denominator(**kwargs)`       | 定义布尔掩码     | `denominator(valid=mask.bool())`         |
| `stat(denominator, **kwargs)` | 带掩码的张量指标 | `stat(loss=tensor, denominator="valid")` |

### 归约类型

使用 `stat()` 时，指标默认为 `AVG_MIN_MAX`，生成三个输出键：

```python
stats_tracker.stat(loss=tensor, denominator="valid")
# 导出：{"loss/avg": 0.5, "loss/min": 0.1, "loss/max": 0.9}
```

可用的归约类型：

| 类型          | 输出                            | 描述             |
| ------------- | ------------------------------- | ---------------- |
| `AVG_MIN_MAX` | `key/avg`, `key/min`, `key/max` | 张量统计的默认值 |
| `AVG`         | `key`                           | 仅加权平均值     |
| `SUM`         | `key`                           | 所有元素求和     |
| `MIN`         | `key`                           | 最小值           |
| `MAX`         | `key`                           | 最大值           |
| `SCALAR`      | `key`, `key__count`             | 用于标量值       |

### 作用域

使用层级作用域组织相关指标：

```python
with stats_tracker.scope("ppo_actor"):
    with stats_tracker.scope("update"):
        stats_tracker.stat(loss=loss_tensor, denominator="valid")
        # 键："ppo_actor/update/loss/avg"
```

### 计时

使用 `timeperf/` 下的自动作用域测量执行时间：

```python
with stats_tracker.record_timing("rollout"):
    batch = actor.prepare_batch(dataloader, workflow)
# 键："timeperf/rollout"
```

### 命名跟踪器

为不同组件隔离指标：

```python
# 训练指标（默认跟踪器）
stats_tracker.scalar(grad_norm=1.5)

# Rollout 指标
stats_tracker.get("rollout").scalar(reward=0.8)

# 评估指标
stats_tracker.get("eval-rollout").scalar(reward=0.9)

# 从所有跟踪器导出
all_stats = stats_tracker.export_all(reduce_group=group)
```

如果设置 `evaluator.eval_before_train: true`，系统会在第一个训练 step 开始前执行一次评估，以便在微调开始前估计初始模型的性能。

## 数据流

从收集到记录的完整指标流程：

```
Rollout 工作器                          训练工作器
───────────────                          ───────────────
workflow.arun_episode()                  actor.ppo_update(batch)
        │                                        │
        ▼                                        ▼
get("rollout").scalar(r=0.5)             stat(tensor, denom=mask)
        │                                        │
        ▼                                        ▼
export_stats(reduce_group=None)          export_stats(reduce_group=dp_group)
{reward: 0.5, reward__count: 1}          → all_reduce 跨 DP rank
        │                                        │
        ▼                                        │
RolloutController.export_stats()                 │
→ 加权平均跨工作器                               │
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
          PPOTrainer._export_and_commit_stats()
                         │
                         ▼
              StatsLogger.commit(stats)
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
          wandb     tensorboard    swanlab
```

## StatsLogger：日志后端

[`StatsLogger`](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_logger.py)
将聚合指标发送到外部日志后端。它由 `PPOTrainer` 自动管理，仅在 rank 0 运行以避免重复日志。

### 支持的后端

| 后端                 | 配置                              | 描述         |
| -------------------- | --------------------------------- | ------------ |
| **Weights & Biases** | `config.stats_logger.wandb`       | 云端实验跟踪 |
| **SwanLab**          | `config.stats_logger.swanlab`     | 替代实验跟踪 |
| **TensorBoard**      | `config.stats_logger.tensorboard` | 本地可视化   |

### 与 PPOTrainer 集成

训练器在每个训练步结束时调用 `StatsLogger.commit()`：

```python
# areal/trainer/rl_trainer.py
def _export_and_commit_stats(self, epoch, epoch_step, global_step):
    # 1. 从所有组件收集指标
    stats = self.actor.export_stats()           # 训练指标（all-reduced）
    stats.update(self.rollout.export_stats())   # Rollout 指标（控制器聚合）
    stats.update(self.eval_rollout.export_stats())  # 评估指标

    # 2. 发送到日志后端（仅 rank 0）
    self.stats_logger.commit(epoch, epoch_step, global_step, stats)
```

### StatsLogger.commit()

`commit()` 方法过滤掉内部计数键并记录到所有配置的后端：

```python
# areal/utils/stats_logger.py
def commit(self, epoch, step, global_step, data):
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # 仅 rank 0 记录

    # 过滤掉 __count 键（用于内部加权平均）
    data = {k: v for k, v in data.items() if not k.endswith("__count")}

    # 记录到所有后端
    wandb.log(data, step=global_step)
    swanlab.log(data, step=global_step)
    if self.summary_writer:
        for key, val in data.items():
            self.summary_writer.add_scalar(key, val, global_step)
```

### 配置

在实验配置中配置日志后端：

```yaml
stats_logger:
  experiment_name: "gsm8k_grpo"
  trial_name: "run_001"
  fileroot: "/path/to/logs"

  wandb:
    mode: "online"  # "online"、"offline" 或 "disabled"
    project: "my-project"
    entity: "my-team"

  swanlab:
    mode: "online"  # "online"、"local" 或 "disabled"
    project: "my-project"

  tensorboard:
    path: "/path/to/tensorboard/logs"  # null 禁用
```

## 训练诊断指标

当 advantage 预处理获得重算的 proximal logprob 时，`ppo_actor/train_infer/` 会在标准 PPO 覆盖原始 rollout
logprob 之前比较两者，不增加模型 forward。 不重算 logprob 的运行和纯 MOPD 预处理路径不会产生这组指标。

设对齐后的有效生成 token 上 `d = log(p_trainer) - log(p_rollout)`：

| 指标                                                                                             | 定义                             |
| ------------------------------------------------------------------------------------------------ | -------------------------------- |
| `logp_diff/{avg,min,max}`                                                                        | 带符号差值 d                     |
| `logp_abs_diff/{avg,min,max}`                                                                    | 绝对差值 abs(d)                  |
| `logp_diff_squared`                                                                              | d² 的 token 加权均值             |
| `trainer_nll`、`rollout_nll`                                                                     | 采样 token 负对数概率的均值      |
| `kl_k1`、`kl_k2`、`kl_k3`                                                                        | -d、d²/2、exp(d)-1-d 的均值      |
| `ratio_outside_1.5`、`ratio_outside_2`、`ratio_outside_3`、`ratio_outside_5`、`ratio_outside_10` | abs(d) > log(阈值) 的 token 比例 |

统计使用后续 rejection/advantage masking 之前的生成掩码。陈旧 rollout 的差异包含策略变化， 不能全部归因于引擎数值误差。KL
估计还取决于采样分布、过滤和 logprob 口径；有限样本的 k1 可以为负，k2 是局部近似。聚合后可以用
`sqrt(max(0, E[d²] - E[abs(d)]²))` 推导绝对差值的总体标准差。

概率比尾部定义与 [R3 论文公式 (3)](https://arxiv.org/html/2510.11370v1) 相同。 当 token 确实来自返回的 rollout
概率分布且支持集匹配时，k1/k3 估计 `KL(rollout || trainer)`。 top-p/top-k 过滤或 greedy 采样后的实际分布，不一定等于接口返回
logprob 所描述的分布。

`nonfinite_logp_fraction` 表示有效 token 中任一侧 logprob 为 NaN 或无穷大的比例。 存在这样的 token 时，概率比尾部指标输出
NaN，不将无效比较误报成零。 掩码之外的 padding 不参与此比例，也不会污染尾部指标。 `kl_k3_overflow_fraction` 记录有限输入差值经
float64 `expm1` 后仍溢出的比例。

`ppo_actor/advantage_{positive,negative,zero}_fraction` 基于塑形后的输入 advantage 和 batch loss
mask，统计位置在 **M2/rejection 过滤之前**。它表示 token 占比，不表示任务成功率或梯度符号。 GSPO 还可能在 loss 内对 advantage
做序列聚合。

`ppo_actor/update/version_stats/sample_staleness_{theta,proximal}_{avg,min,max}`
仅统计掩码有效且版本非负的生成 token。均值按 token 加权，极值是聚合组中的全局极值。 theta/proximal 两组兼容字段均使用最近发布的
checkpoint 版本：actor 和 rollout 在 PPO 更新后一起升版， 这不是 optimizer minibatch
计数。`stale_token_fraction` 表示 behavior 版本落后的比例， `future_token_fraction` 表示 behavior
版本超前的比例。包括 pure MOPD 在内，诊断在 `_ppo_update` 中局部左移 raw rollout version 到预测位置， 不修改 loss
的原有近端策略近似。统计总体在 **M2/rejection 过滤之前**，每次 `_ppo_update` 计一次，与 minibatch forward 次数无关。
`n_valid_generated_tokens` 累加这部分 token 观测；不同 update 重用同一轨迹仍会重复计数。 没有有效 token
时省略均值和极值，不填充为零。

新增诊断使用 `stat_compact`：每个指标在每个 batch 仅保留四个 float64 标量（掩码和、数量、最小、最大） 直到 export，避免原先约 106
字节/padded token 的诊断张量长期留存。已有 PPO 指标仍有自身的全量张量， 诊断计算过程中也会临时分配逐 token
张量。长上下文的峰值内存和运行时间仍需按实际 workload 验证。

可选的 loglinear 近端策略近似仍读取原始 token 位置的 rollout version，且保留原先 `current_version - 1`
的插值假设。其算法语义需要另行按真实 optimizer step 验证； 这里修正的 checkpoint 年龄指标不能证明 loglinear 路径也已修复。

## 最佳实践

1. **选择正确的范式**：对标量使用 `scalar()`，对批量 PyTorch 张量（通常是训练指标）使用带分母的 `stat()`。

1. **先定义分母**：始终在 `stat()` 之前调用 `denominator()` 来建立掩码关系。

1. **使用命名跟踪器**：使用 `stats_tracker.get(workflow_context.stat_scope()).scalar(...)` 将
   Rollout（`"rollout"`）和评估（`"eval-rollout"`）指标与训练指标隔离。
