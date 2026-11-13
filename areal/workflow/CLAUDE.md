[根目录](../../CLAUDE.md) > **areal/workflow**

# areal.workflow - Rollout 工作流实现

## 变更记录 (Changelog)

### 2026-01-31 - 初始化

- 模块文档创建
- 识别 9 个工作流实现文件

---

## 模块职责

实现各种 Rollout 工作流，用于生成训练数据：

- **多轮对话**：`MultiTurnWorkflow` - 多次尝试直到奖励为正
- **RLVR**：`RLVRWorkflow` - 单轮推理与验证
- **视觉 RLVR**：`VisionRLVRWorkflow` - 支持视觉输入的 RLVR
- **Agent 集成**：OpenAI、Anthropic、LangChain、OpenAI Agents SDK

## 入口与启动

工作流通过 `areal.core.workflow_executor.WorkflowExecutor` 调用，不直接启动。

## 对外接口

### 核心工作流

| 类名                   | 文件                | 职责                                   |
| ---------------------- | ------------------- | -------------------------------------- |
| `MultiTurnWorkflow`    | `multi_turn.py`     | 多轮对话，直到奖励为正或达到最大轮数   |
| `RLVRWorkflow`         | `rlvr.py`           | 单轮推理与验证（RLVR 算法）            |
| `VisionRLVRWorkflow`   | `vision_rlvr.py`    | 支持视觉输入的 RLVR                    |

### Agent 集成工作流

| 类名                      | 文件                            | 职责                                   |
| ------------------------- | ------------------------------- | -------------------------------------- |
| `OpenAIMathAgent`         | `openai/math_agent.py`          | OpenAI SDK 数学 Agent                  |
| `AnthropicMathAgent`      | `anthropic/math_agent.py`       | Anthropic SDK 数学 Agent               |
| `LangChainMathAgent`      | `langchain/math_agent.py`       | LangChain 数学 Agent                   |
| `OpenAIAgentMathAgent`    | `openai_agent/math_agent.py`    | OpenAI Agents SDK 数学 Agent           |

## 关键依赖与配置

### 外部依赖

- `torch`：张量操作
- `transformers`：Tokenizer
- `openai`：OpenAI SDK（Agent 集成）
- `anthropic`：Anthropic SDK（Agent 集成）
- `langchain`：LangChain（Agent 集成）

### 内部依赖

- `areal.api.workflow_api`：`RolloutWorkflow`、`AgentWorkflow` 基类
- `areal.api.engine_api`：`InferenceEngine`
- `areal.api.reward_api`：`AsyncRewardWrapper`
- `areal.api.cli_args`：`GenerationHyperparameters`
- `areal.utils.logging`：日志工具

## 数据模型

### MultiTurnWorkflow 输入输出

**输入**（`data: dict`）：

```python
{
    "messages": [{"role": "user", "content": "问题"}],
    # 其他奖励函数需要的字段
}
```

**输出**（`dict`）：

```python
{
    "seq": [token_ids],           # 完整序列（prompt + completions）
    "logprobs": [logprobs],       # 对数概率
    "loss_mask": [0/1],           # 损失掩码（仅 completion 部分为 1）
    "versions": [version_ids],    # 模型版本 ID
    "reward": float,              # 最终奖励
    "prompt_str": str,            # Prompt 字符串
    "completions_str": str,       # Completion 字符串
}
```

### RLVRWorkflow 输入输出

**输入**：同 `MultiTurnWorkflow`

**输出**：

```python
{
    "seq": [token_ids],
    "logprobs": [logprobs],
    "loss_mask": [0/1],
    "versions": [version_ids],
    "reward": float,
    "prompt_str": str,
    "completions_str": str,
    "n_samples": int,             # 采样数量
}
```

## 测试与质量

- **测试覆盖**：无专门的单元测试，通过集成测试（`areal/tests/grpo/`, `areal/tests/sft/`）验证
- **质量工具**：Ruff（格式化与 lint）

## 常见问题 (FAQ)

### Q: 如何实现自定义 Workflow？

A: 继承 `RolloutWorkflow`，实现 `arun_episode` 方法：

```python
from areal.api.workflow_api import RolloutWorkflow
from areal.api.engine_api import InferenceEngine

class MyWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data: dict):
        # 1. 构造 ModelRequest
        # 2. 调用 engine.agenerate()
        # 3. 计算奖励
        # 4. 返回结果字典
        return {"seq": ..., "logprobs": ..., "reward": ...}
```

### Q: MultiTurnWorkflow 如何处理多轮对话？

A: 每轮生成后，如果奖励为 0（错误），则在 prompt 后追加错误提示和上一轮的 completion，继续生成。最多尝试 `max_turns` 轮。

### Q: 如何集成外部 Agent SDK？

A: 继承 `AgentWorkflow`，实现 `run` 方法，使用 `extra_kwargs["base_url"]` 和 `extra_kwargs["http_client"]` 连接 AReaL 的 OpenAI 兼容代理服务器。参考 `openai/math_agent.py`。

## 相关文件清单

```
areal/workflow/
├── multi_turn.py              # 多轮对话工作流（核心）
├── rlvr.py                    # RLVR 工作流（核心）
├── vision_rlvr.py             # 视觉 RLVR 工作流
├── openai/
│   └── math_agent.py          # OpenAI SDK 数学 Agent
├── anthropic/
│   ├── __init__.py
│   └── math_agent.py          # Anthropic SDK 数学 Agent
├── langchain/
│   ├── __init__.py
│   └── math_agent.py          # LangChain 数学 Agent
└── openai_agent/
    └── math_agent.py          # OpenAI Agents SDK 数学 Agent
```

## 下一步建议

- 添加 Workflow 的单元测试
- 补充 Agent 集成的文档与示例
- 优化多轮对话的 prompt 构造逻辑
