[根目录](../../CLAUDE.md) > **areal/api**

# areal.api - API 与配置契约

## 变更记录 (Changelog)

### 2026-01-31 - 初始化

- 模块文档创建
- 识别 9 个核心文件

---

## 模块职责

定义 AReaL 系统的核心 API 契约与配置数据类：

- **配置数据类**：CLI 参数、训练/推理超参数、并行策略
- **引擎契约**：`TrainEngine`、`InferenceEngine` 抽象基类
- **工作流契约**：`RolloutWorkflow`、`AgentWorkflow` 抽象基类
- **奖励契约**：`RewardFunction` 与异步包装器
- **调度器契约**：`Scheduler` 抽象基类
- **数据结构**：`ModelRequest`、`ModelResponse`、`ParamSpec` 等

## 入口与启动

无独立启动入口，作为其他模块的依赖被导入。

## 对外接口

### 核心抽象类

| 类名                | 文件                 | 职责                                   |
| ------------------- | -------------------- | -------------------------------------- |
| `TrainEngine`       | `engine_api.py`      | 训练引擎抽象基类（FSDP/Megatron）      |
| `InferenceEngine`   | `engine_api.py`      | 推理引擎抽象基类（SGLang/vLLM）        |
| `RolloutWorkflow`   | `workflow_api.py`    | Rollout 工作流抽象基类                 |
| `AgentWorkflow`     | `workflow_api.py`    | Agent 工作流抽象基类（OpenAI SDK 集成）|
| `RewardFunction`    | `reward_api.py`      | 奖励函数抽象基类                       |
| `Scheduler`         | `scheduler_api.py`   | 调度器抽象基类（Local/Ray/Slurm）      |

### 配置数据类

| 类名                          | 文件            | 职责                                   |
| ----------------------------- | --------------- | -------------------------------------- |
| `NormConfig`                  | `cli_args.py`   | 奖励/优势归一化配置                    |
| `MicroBatchSpec`              | `cli_args.py`   | 微批次划分规格                         |
| `GenerationHyperparameters`   | `cli_args.py`   | 生成超参数（温度、top_p、max_tokens）  |
| `TrainingHyperparameters`     | `cli_args.py`   | 训练超参数（学习率、优化器、调度器）    |
| `ParallelStrategy`            | `alloc_mode.py` | 并行策略（DP/TP/PP/CP/EP）             |

### 数据结构

| 类名                | 文件            | 职责                                   |
| ------------------- | --------------- | -------------------------------------- |
| `ModelRequest`      | `io_struct.py`  | 推理请求（input_ids、gconfig）         |
| `ModelResponse`     | `io_struct.py`  | 推理响应（output_tokens、logprobs）    |
| `ParamSpec`         | `io_struct.py`  | 参数规格（形状、dtype、设备）          |
| `WeightUpdateMeta`  | `io_struct.py`  | 权重更新元数据                         |
| `SaveLoadMeta`      | `io_struct.py`  | 检查点保存/加载元数据                  |

## 关键依赖与配置

### 外部依赖

- `torch`：张量操作与分布式通信
- `transformers`：Tokenizer 与模型配置
- `omegaconf`、`hydra-core`：配置管理
- `pydantic`：数据验证

### 内部依赖

- `areal.utils.logging`：日志工具
- `areal.utils.name_resolve`：动态导入

## 数据模型

### CLI 参数结构

```python
@dataclass
class TrainingHyperparameters:
    # 优化器
    optimizer: str = "adam"
    lr: float = 1e-5
    weight_decay: float = 0.0

    # 学习率调度
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0

    # 训练配置
    n_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
```

### 并行策略

```python
@dataclass
class ParallelStrategy:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
```

## 测试与质量

- **测试覆盖**：部分配置类有单元测试（`areal/tests/test_adv_norm_config.py`、`test_allocation_mode.py`）
- **质量工具**：Ruff（格式化与 lint）、pre-commit hooks

## 常见问题 (FAQ)

### Q: 如何添加新的配置字段？

A: 在对应的 `@dataclass` 中添加字段，并提供默认值。注意向后兼容性。

### Q: 如何实现自定义 Workflow？

A: 继承 `RolloutWorkflow` 或 `AgentWorkflow`，实现 `arun_episode` 或 `run` 方法。参考 `areal/workflow/multi_turn.py`。

### Q: 如何实现自定义 Engine？

A: 继承 `TrainEngine` 或 `InferenceEngine`，实现所有抽象方法。参考 `areal/engine/fsdp_engine.py`。

## 相关文件清单

```
areal/api/
├── __init__.py
├── alloc_mode.py          # 并行策略与分配模式
├── cli_args.py            # CLI 参数与配置数据类（核心）
├── engine_api.py          # 引擎抽象基类（核心）
├── env_api.py             # 环境 API（实验性）
├── io_struct.py           # 数据结构定义
├── reward_api.py          # 奖励函数抽象基类
├── scheduler_api.py       # 调度器抽象基类
└── workflow_api.py        # 工作流抽象基类（核心）
```

## 下一步建议

- 补充 `cli_args.py` 中各配置类的详细文档
- 添加配置验证的单元测试
- 完善 `env_api.py` 的实验性功能
