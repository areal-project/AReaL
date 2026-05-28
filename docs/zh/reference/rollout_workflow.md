# RolloutWorkflow 参考

本文档描述 `RolloutWorkflow` 抽象，它是 AReaL 强化学习流水线中实现 Rollout 生成的核心接口。

**注意：**

1. 本页面向希望深入理解代码库的开发者。对于代理 RL 训练，请使用 [Agentic RL Guide](../tutorial/agentic_rl.md) 中描述的高级
   API。

1. **遗留模式**：直接子类化 `RolloutWorkflow` 被视为遗留方式，不应主动使用。对于新的代理 RL 工作流，请使用带有 `async def run()`
   的[代理工作流模式](./agent_workflow.md)。

## 概述

`RolloutWorkflow` 定义如何从输入数据生成训练轨迹。它封装了以下逻辑：

- 对提示进行分词并准备模型输入
- 调用推理引擎生成回复
- 为生成的输出计算奖励
- 将结果打包成用于训练的张量字典

## 接口

```python
from areal.api.workflow_api import RolloutWorkflow

class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """运行单个工作流 episode。"""
        ...
```

### 参数

| 参数     | 类型              | 描述                       |
| -------- | ----------------- | -------------------------- |
| `engine` | `InferenceEngine` | 用于生成模型回复的推理引擎 |
| `data`   | `dict[str, Any]`  | 数据加载器的单个样本       |

### 返回类型

`arun_episode` 方法支持三种返回类型：

| 返回类型                                    | 描述                                                              |
| ------------------------------------------- | ----------------------------------------------------------------- |
| `dict[str, torch.Tensor]`                   | 用于训练的标准张量格式                                            |
| `dict[str, InteractionWithTokenLogpReward]` | Token 级别的交互（自动转换为张量）；由高级 `ArealOpenAI` API 生成 |
| `None`                                      | 拒绝的轨迹，排除在训练之外                                        |

## 张量字典格式

返回张量字典时，预期包含以下字段：

| 字段             | 形状                    | 类型    | 必需 | 描述                            |
| ---------------- | ----------------------- | ------- | ---- | ------------------------------- |
| `input_ids`      | `[batch_size, seq_len]` | int32   | 是   | Token ID（提示 + 生成内容）     |
| `attention_mask` | `[batch_size, seq_len]` | bool    | 是   | 有效 token 掩码                 |
| `loss_mask`      | `[batch_size, seq_len]` | int32   | 否   | 生成内容 token 掩码（1 = 训练） |
| `logprobs`       | `[batch_size, seq_len]` | float32 | 否   | 每个 token 的对数概率           |
| `rewards`        | `[batch_size]`          | float32 | 否   | 每序列奖励                      |
| `versions`       | `[batch_size, seq_len]` | int32   | 否   | 生成 token 时的权重版本         |

返回值示例：

```python
return {
    "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
    "attention_mask": torch.ones(1, 5, dtype=torch.bool),
    "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
    "logprobs": torch.tensor([[0.0, 0.0, -0.5, -0.3, -0.2]], dtype=torch.float32),
    "rewards": torch.tensor([1.0], dtype=torch.float32),
    "versions": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
}
```

## 工作流上下文

在 `arun_episode` 中，通过 `workflow_context` 模块访问执行上下文。每个工作流实例有其自己的隔离上下文：

```python
from areal.infra import workflow_context

async def arun_episode(self, engine, data):
    # 获取当前执行上下文
    ctx = workflow_context.get()

    # 检查是否在评估模式运行
    if ctx.is_eval:
        # 使用不同参数进行评估
        ...

    # 获取用于日志的任务 ID
    task_id = ctx.task_id

    # 根据模式获取统计作用域（"rollout" 或 "eval-rollout"）
    scope = workflow_context.stat_scope()
```

## 轨迹转储

当 `InferenceEngineConfig.dump_to_file=True` 时，轨迹自动保存到磁盘用于调试和分析。

### 配置

```yaml
rollout:
  dump_to_file: true
  fileroot: "/path/to/logs"
  tokenizer_path: "model/tokenizer" # 文本解码必需
```

### 输出位置

轨迹保存到：

```
{fileroot}/{experiment_name}/{trial_name}/[rollout|eval-rollout]/{version}/{task_id}.jsonl
```

示例：

```
/tmp/areal/my_exp/trial1/rollout/5/42.jsonl
```

### 输出格式

JSONL 文件的每一行包含：

```json
{
  "task_id": 42,
  "sample_idx": 0,
  "seqlen": 256,
  "prompt_len": 128,
  "head_version": 5,
  "tail_version": 5,
  "reward": 1.0,
  "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "completion": "The answer is 4.<|im_end|>"
}
```

## 分组 Rollout

分组 Rollout 对每个输入提示多次运行相同工作流，生成多样化回复用于训练。这对于像 GRPO 这样受益于每个提示多个样本的算法很有用。

### 配置

提交 Rollout 时设置 `group_size`：

```python
engine.submit(
    data=sample,
    workflow=MyWorkflow,
    workflow_kwargs={...},
    group_size=4,  # 每个输入运行工作流 4 次
)
```

或通过 CLI：

```yaml
rollout:
  group_size: 4
```

### 工作原理

当 `group_size > 1` 时，工作流被包装在 `GroupedRolloutWorkflow` 中：

1. 包装器使用 `asyncio.gather` 并发运行 `arun_episode` `group_size` 次
1. 根据类型合并结果：
   - **张量字典**：沿批次维度连接
   - **InteractionWithTokenLogpReward 字典**：合并为单个字典
1. 如果某些运行返回 `None`（拒绝），仅保留有效结果
1. 如果所有运行都返回 `None`，则整个分组结果为 `None`

### 输出形状

当 `group_size=4` 且工作流返回 `[1, seq_len]` 张量时，分组输出的形状为 `[4, seq_len]`（4 个样本连接）。

### 实现

来自 `areal/infra/remote_inf_engine.py`：

```python
class GroupedRolloutWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 并发运行 N 次
        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data)
              for _ in range(self.group_size)]
        )

        # 过滤 None 结果
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return None

        # 根据结果类型合并
        if all_interaction_dicts(valid_results):
            return merge_dicts(valid_results)
        else:
            return concat_padded_tensors(valid_results)
```

## 实现自定义工作流

创建自定义工作流：

1. **子类化 `RolloutWorkflow`**：

```python
from areal.api.workflow_api import RolloutWorkflow

class MyWorkflow(RolloutWorkflow):
    def __init__(self, tokenizer, gconfig, **kwargs):
        self.tokenizer = tokenizer
        self.gconfig = gconfig

    async def arun_episode(self, engine, data):
        # 1. 准备输入
        input_ids = self.tokenizer.encode(data["prompt"])

        # 2. 生成回复
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig,
            tokenizer=self.tokenizer,
        )
        resp = await engine.agenerate(req)

        # 3. 计算奖励
        reward = self.compute_reward(resp, data)

        # 4. 返回张量字典（或 None 拒绝）
        if reward < 0:
            return None

        return self.build_tensor_dict(resp, reward)
```

2. **向训练器注册**：

```python
trainer.train(
    workflow=MyWorkflow,
    workflow_kwargs={
        "tokenizer": tokenizer,
        "gconfig": config.gconfig,
    },
)
```

## 工作流解析

工作流可以通过多种方式指定：

| 格式       | 示例                          | 描述              |
| ---------- | ----------------------------- | ----------------- |
| 实例       | `MyWorkflow(...)`             | 预实例化的工作流  |
| 类         | `MyWorkflow`                  | 类（需要 kwargs） |
| 字符串路径 | `"my_module.MyWorkflow"`      | 动态导入          |
| 代理工作流 | 任何带 `async def run()` 的类 | 带代理支持封装    |

训练系统自动将这些解析为 `RolloutWorkflow` 实例。

## 另见

- [Agentic RL Tutorial](../tutorial/agentic_rl.md) - 使用代理框架训练
- [Adding Custom Workflows](../customization/agent.md) - 分步指南
