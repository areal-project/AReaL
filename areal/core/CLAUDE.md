[根目录](../../CLAUDE.md) > **areal/core**

# areal.core - 核心运行时

## 变更记录 (Changelog)

### 2026-01-31 - 初始化

- 模块文档创建
- 识别 6 个核心运行时组件

---

## 模块职责

实现 AReaL 的核心运行时组件：

- **工作流执行器**：`WorkflowExecutor` - 异步执行 Rollout 工作流
- **分布式 Rollout**：`DistRollout` - 分布式数据生成
- **异步任务运行器**：`AsyncTaskRunner` - 管理异步任务队列
- **远程推理引擎**：`RemoteInfEngine` - 远程推理引擎代理
- **陈旧度管理器**：`StalenessManager` - 管理模型版本陈旧度
- **工作流上下文**：`workflow_context` - 全局上下文管理

## 入口与启动

核心运行时由 `areal.controller` 调用，不直接启动。

## 对外接口

### 核心类

| 类名                  | 文件                      | 职责                                   |
| --------------------- | ------------------------- | -------------------------------------- |
| `WorkflowExecutor`    | `workflow_executor.py`    | 异步执行 Rollout 工作流（核心）        |
| `DistRollout`         | `dist_rollout.py`         | 分布式数据生成（核心）                 |
| `AsyncTaskRunner`     | `async_task_runner.py`    | 异步任务队列管理                       |
| `RemoteInfEngine`     | `remote_inf_engine.py`    | 远程推理引擎代理                       |
| `StalenessManager`    | `staleness_manager.py`    | 模型版本陈旧度管理                     |

### 全局上下文

| 函数/变量                     | 文件                      | 职责                                   |
| ----------------------------- | ------------------------- | -------------------------------------- |
| `workflow_context.get()`      | `workflow_context.py`     | 获取当前工作流上下文                   |
| `workflow_context.set()`      | `workflow_context.py`     | 设置当前工作流上下文                   |

## 关键依赖与配置

### 外部依赖

- `torch`：张量操作与分布式通信
- `asyncio`：异步编程
- `aiohttp`：异步 HTTP 客户端
- `pyzmq`：ZeroMQ 消息队列

### 内部依赖

- `areal.api.workflow_api`：`RolloutWorkflow`、`AgentWorkflow`
- `areal.api.engine_api`：`InferenceEngine`
- `areal.api.cli_args`：配置数据类
- `areal.utils.logging`：日志工具
- `areal.utils.distributed`：分布式工具

## 数据模型

### WorkflowExecutor 初始化参数

```python
WorkflowExecutor(
    workflow: RolloutWorkflow | AgentWorkflow,  # 工作流实例
    engine: InferenceEngine,                    # 推理引擎
    n_workers: int = 1,                         # 并发 worker 数量
    max_queue_size: int = 100,                  # 最大队列大小
)
```

### DistRollout 初始化参数

```python
DistRollout(
    workflow_executor: WorkflowExecutor,        # 工作流执行器
    dataloader: DataLoader,                     # 数据加载器
    n_samples_per_prompt: int = 1,              # 每个 prompt 的采样数
    staleness_coef: float = 0.0,                # 陈旧度系数
)
```

### StalenessManager 数据结构

```python
{
    "current_version": int,                     # 当前模型版本
    "version_timestamps": Dict[int, float],     # 版本时间戳
    "staleness_coef": float,                    # 陈旧度系数
}
```

## 测试与质量

- **测试覆盖**：
  - `areal/tests/test_async_task_runner.py`
  - `areal/tests/test_staleness_manager.py`
  - `areal/tests/test_rollout_controller.py`（集成测试）
- **质量工具**：Ruff、pre-commit hooks

## 常见问题 (FAQ)

### Q: WorkflowExecutor 如何处理并发？

A: 使用 `asyncio.Queue` 管理任务队列，启动 `n_workers` 个异步 worker 并发执行工作流。

### Q: 什么是陈旧度（Staleness）？

A: 在异步 RL 训练中，推理使用的模型版本可能落后于训练版本。陈旧度系数用于调整旧版本数据的权重。

### Q: 如何调试分布式 Rollout？

A: 参考 `docs/best_practices/debugging.md`：
- 检查日志中的 `DistRollout` 和 `WorkflowExecutor` 输出
- 使用 `workflow_context.get()` 获取当前上下文信息
- 启用 `AREAL_DEBUG=1` 环境变量

### Q: RemoteInfEngine 与本地 InferenceEngine 的区别？

A: `RemoteInfEngine` 通过 HTTP/ZMQ 与远程推理服务器通信，适合推理与训练分离的场景。本地 `InferenceEngine` 直接调用推理引擎。

## 相关文件清单

```
areal/core/
├── __init__.py
├── workflow_executor.py       # 工作流执行器（核心）
├── dist_rollout.py            # 分布式 Rollout（核心）
├── async_task_runner.py       # 异步任务运行器
├── remote_inf_engine.py       # 远程推理引擎代理
├── staleness_manager.py       # 陈旧度管理器
└── workflow_context.py        # 工作流上下文
```

## 下一步建议

- 补充 WorkflowExecutor 的性能调优文档
- 添加 DistRollout 的分布式测试
- 优化 AsyncTaskRunner 的队列管理逻辑
