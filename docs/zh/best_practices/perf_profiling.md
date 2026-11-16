# 性能分析

AReaL 通过 `perf_tracer` 提供轻量级分析基础设施，帮助您识别分布式训练 Workflow 中的性能瓶颈。追踪器发出与 Chrome Trace
兼容的事件，可以在 Perfetto 或 chrome://tracing 中可视化，便于跨多个 ranks 关联计算、通信和 I/O。

**关键功能**：

- 灵活的追踪
  API：装饰器（`@trace_perf`、`@session_context`、`@trace_session`）、上下文管理器（`trace_scope`/`atrace_scope`、`trace_session_phase`/`atrace_session_phase`）和标记（`instant`）
- **每个会话的生命周期追踪**（任务注册 → 会话创建 → 生成 → 奖励 → 最终化），包含派生指标（总时间、生成时间、工具调用时间、奖励计算时间）
- **任务-会话层级结构**，用于追踪数据集级别的任务及其样本级别的会话

## 快速开始

### 1. 在配置中启用追踪

在训练脚本的 YAML 配置或 CLI 覆盖中添加 {ref}`PerfTracerConfig <section-perf-tracer>`：

```yaml
perf_tracer:
  enabled: true
  experiment_name: ${experiment_name}  # 重用顶层元数据
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}        # 共享文件系统路径
  save_interval: 1                     # 每步写入追踪
  session_tracer:
    enabled: true                      # 追踪每个会话的生命周期
    flush_threshold: 100               # 缓冲 100 个会话后再刷新
```

在 YAML 配置文件中配置追踪器。

### 2. 初始化追踪器

在启动时为每个 rank 调用一次 `perf_tracer.configure()`：

```python
from areal.utils import perf_tracer

if config.perf_tracer is not None:
    perf_tracer.configure(config.perf_tracer, rank=rank)
```

全局追踪器现在对该进程处于活动状态。

### 3. 运行训练并收集追踪

像往常一样执行训练脚本。追踪器自动将事件写入 `fileroot/logs/.../perf_tracer/traces-r{rank}.jsonl`。对于多 rank
作业，每个 rank 产生自己的文件。

```bash
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local
```

### 4. 在 Perfetto 中查看追踪

将 JSONL 转换为 JSON 并在 [Perfetto](https://ui.perfetto.dev/) 或 chrome://tracing 中打开：

```bash
python -m areal.tools.perf_trace_converter logs/**/perf_tracer/traces-*.jsonl merged.json
```

## 分析模式和 API

### 模式 1：使用 `@trace_perf` 追踪整个函数

**用例**：了解关键方法（train_batch、forward、ppo_update 等）花费的总时间。

**API**：`@trace_perf(name, category=...)`

- 包装同步/异步函数的装饰器
- 自动记录开始/结束时间戳
- 优雅地处理异常

**示例**（来自 `areal/engine/fsdp_engine.py`）：

```python
from areal.utils.perf_tracer import trace_perf

@trace_perf("fsdp_engine.train_batch")
def train_batch(self, input_: dict[str, Any], loss_fn, loss_weight_fn):
    # Training logic here
    ...
```

这会在 Chrome Trace 输出中创建一个名为 `fsdp_engine.train_batch` 的"完整事件"（持续时间跨度）。

### 模式 2：使用 `trace_scope` / `atrace_scope` 追踪代码块

**用例**：分析特定代码段，而无需将它们提取到方法中。

**API**：同步/异步代码的上下文管理器

- `with trace_scope(name, category, args)`：同步上下文
- `async with atrace_scope(name, category, args)`：异步上下文

**示例**：

```python
from areal.utils import perf_tracer
from areal.utils.perf_tracer import Category

with perf_tracer.trace_scope(
    "train.rollout",
    args={"global_step": global_step, "epoch_step": step},
):
    batch = actor.prepare_batch(dataloader, n_samples)
    # Rollout generation happens here
```

`args` 字典将元数据（步骤编号、批大小等）附加到事件上，可在追踪查看器中查看。

### 模式 3：使用 `@trace_session` 追踪会话生命周期

**用例**：测量异步 rollout Workflow 的每个会话时间（例如，单个提示从提交到奖励计算需要多长时间？）。

**API**：`@trace_session(phase)`

- 参与会话处理的同步/异步方法的装饰器
- 自动读取由 `@session_context()` 填充的 `session_id`
- 记录 `mark_{phase}_start` 和 `mark_{phase}_end` 事件

**示例**（来自 `areal/workflow/rlvr.py`）：

```python
from areal import workflow_context
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_perf,
    trace_session,
)

class RLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        # WorkflowExecutor automatically sets task_id before calling this method
        # Each sample will register its own session_id

        # Generate responses and collect rewards for n_samples
        resp, reward = self._collect_samples(engine, req, prompt_str, data)

        # Build result tensors
        return self._build_result_tensors(resp, reward)

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float, str]:
        """Generate one sample and compute its reward with session tracing."""
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(
            resp,
            prompt_str,
            task_data,
        )

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    @trace_session("reward")
    async def _compute_rewards(self, resp, prompt_str, task_data):
        """Compute rewards with automatic phase tracing."""
        completion_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str, completion_str,
            resp.input_tokens, resp.output_tokens,
            **task_data
        )
        return reward
```

#### `session_context` 装饰器

`session_context()` 在当前上下文中已存在 `task_id` 时，为每次调用注册一个新会话。在处理单个样本的 Workflow
方法上使用它，以便下游辅助函数（例如 `atrace_session_phase`、`@trace_session`）可以从 contextvars 中读取
`session_id`。

```python
from areal.utils.perf_tracer import atrace_session_phase, session_context

class MiniWorkflow:
    @session_context()
    async def collect(self, engine, request):
        async with atrace_session_phase("generate"):
            return await engine.agenerate(request)
```

**工作原理**：

1. `WorkflowExecutor` 在调用 `arun_episode` 之前调用 `perf_tracer.set_task_id()`
1. `arun_episode` 生成多个 `_collect_samples` 调用（每个样本一个）
1. 每个 `_collect_samples` 应用 `@perf_tracer.session_context()` 装饰器来自动注册会话并将 `task_id` /
   `session_id` 放入上下文变量
1. 上下文变量透明地存储活动任务/会话 ID
1. 子异步函数自动继承此上下文
1. `@trace_session("reward")` 读取会话 ID 并记录阶段开始/结束事件
1. 会话追踪出现在 `session_tracer/sessions-r{rank}.jsonl` 中，包含计算出的指标如 `reward_s`、`generate_s`

### 模式 4：使用 `atrace_session_phase` 和 `trace_session_phase` 的手动阶段作用域

**用例**：追踪无法干净地提取到方法中的阶段（例如内联生成循环或后处理），同时重用模式 3 中创建的会话上下文。

**API**：

```python
async with atrace_session_phase(
    phase,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
):
    ...

with trace_session_phase(
    phase,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
):
    ...
```

- 会话阶段追踪的上下文管理器（异步和同步变体）
- 自动发出 `mark_{phase}_start` / `mark_{phase}_end` 事件，并附加可选负载以获取更丰富的追踪元数据

**示例**（继续模式 3 中的 `session_context` Workflow）：

```python
@session_context()
async def _collect_samples(..., n_attempts: int = 1):
    async with perf_tracer.atrace_session_phase(
        "generate",
        start_payload={"attempts": n_attempts},
    ):
        response = await engine.agenerate(req)

    reward, completion_str = await self._compute_rewards(response, prompt_str, data)

    with perf_tracer.trace_session_phase(
        "postprocess",
        end_payload={"accepted": response.accepted},
    ):
        filtered = self._postprocess(response)

    return filtered, reward, completion_str
```

异步作用域发出 `engine.agenerate` 的计时，而同步作用域涵盖任何 CPU 端后处理。两者都共享由 `@session_context()` 装饰器隐式提供的
`session_id`，因此它们的事件出现在同一会话追踪记录中。

### 模式 5：使用 `instant()` 添加即时标记

**用例**：标记特定时间点（例如"批次已准备"、"队列状态快照"）。

**API**：`perf_tracer.instant(name, category, args)`

- 创建时间点标记（不是持续时间）
- 对没有有意义持续时间的事件很有用

**示例**（来自 `areal/infra/workflow_executor.py`）：

```python
perf_tracer.instant(
    "workflow_executor.prepare_batch",
    category="scheduler",
    args={"data": len(data)}
)

perf_tracer.instant(
    "workflow_executor.wait",
    category="scheduler",
    args={
        "queue_size": runner.get_output_queue_size(),
        "pending_results": len(pending_results),
    }
)
```

### 模式 6：使用 `trace_session_event` 手动会话生命周期事件

**用例**：在编排级别追踪会话生命周期（提交、执行、消费）。

**API**：
`perf_tracer.trace_session_event(method, session_id=None, task_id=None, **payload)`

- 手动记录会话追踪的生命周期事件
- 由 `WorkflowExecutor` 用于追踪完整会话生命周期
- 事件：`mark_finalized`，以及通过 `@trace_session` 装饰器的阶段事件
- 参数：使用 `session_id=` 定位特定会话，或使用 `task_id=` 定位任务中的所有会话

**示例**（来自 `areal/infra/workflow_executor.py`）：

```python
from areal.utils.perf_tracer import trace_session_event

# Run workflow
traj = await workflow.arun_episode(engine, data)

# Mark execution end with status
if should_accept:
    trace_session_event(
        "mark_finalized",
        session_id=session_id,
        status="accepted",
    )
else:
    trace_session_event(
        "mark_finalized",
        session_id=session_id,
        status="rejected",
        reason="stale_weight",
    )
```

## 会话生命周期追踪

启用 `perf_tracer.session_tracer.enabled=true` 以追踪每个会话的指标，而不仅仅是性能跨度。这对于诊断队列问题和过期问题很有用。

### 理解任务-会话层级结构

AReaL 的会话追踪器使用两级层级结构来追踪 rollout 执行：

- **Task**（数据集级别）：代表数据集中的一个数据点。每次 `arun_episode` 调用注册一次。
- **Session**（样本级别）：代表一个生成的样本。当 `n_samples > 1` 时，一个任务生成多个会话。

**示例**：如果您的配置设置 `n_samples=4`，每个数据集项创建：

- 1 个任务（由 `WorkflowExecutor` 注册）
- 4 个会话（在 `_collect_samples` 中注册，每个生成一个）

此层级结构支持：

- 追踪每个样本的指标（生成时间、奖励）
- 聚合每个数据集项的统计信息（跨样本的接受率）
- 调试任务内哪些特定样本失败或被拒绝

### 追踪内容

每个会话记录包含：

- **任务/会话层级**：`task_id`（数据集级别）、`session_id`（样本级别）
- **生命周期时间戳**：`submit_ts`、`finalized_ts`
- **状态**：`pending`、`accepted`、`rejected`、`failed`、`dropped`
- **阶段**：`generate`、`reward`、`toolcall` 的多次执行，包含开始/结束时间
- **派生指标**：`total_s`、`generate_s`、`reward_s`、`toolcall_s`
- **上下文**：`reason`（可选，用于拒绝/失败的会话）

### 输出格式

会话追踪写入 `session_tracer/sessions-r{rank}.jsonl`。每行是一个 JSON 对象：

```json
{
    "task_id": 23,
    "session_id": 93,
    "rank": 0,
    "status": "accepted",
    "submit_ts": 7939251.674969524,
    "finalized_ts": 7939254.632833603,
    "total_s": 2.957864078693092,
    "generate_s": 2.65427936706692,
    "reward_s": 0.133724981918931,
    "toolcall_s": 0.156789012345678,
    "phases": {
        "generate": [
            {
                "start_ts": 7939251.674977085,
                "end_ts": 7939254.329256452
            }
        ],
        "reward": [
            {
                "start_ts": 7939254.32926108,
                "end_ts": 7939254.462986062
            }
        ],
        "toolcall": [
            {
                "start_ts": 7939254.463123456,
                "end_ts": 7939254.619912468
            }
        ]
    }
}
```

### 添加自定义阶段

引入新阶段（例如验证通道、安全过滤器）的 Workflow 可以通过少量扩展 `areal/utils/perf_tracer.py` 来发出专用的阶段跨度：

1. 在 `SessionTraceEvent` 中声明开始/结束事件：

   ```python
   class SessionTraceEvent(str, Enum):
        VALIDATION_START = "validation_start"
        VALIDATION_END = "validation_end"
   ```

1. 通过在 `SessionRecord.default_phase_configs()` 中追加 `PhaseSpec` 来注册阶段（根据需要设置
   `allow_multiple` 或 `ready_on_complete`）。

1. 如果您希望 JSONL 输出中有 `validation_s`，请添加一个辅助函数如 `_compute_validation_time()` 和在
   `SessionRecord.default_field_spec()` 中匹配的 `FieldSpec`。

1. 通过更新 `_SESSION_TRACE_METHOD_TO_EVENT` 将 `"mark_validation_start"` 和
   `"mark_validation_end"` 指向新的枚举条目来映射追踪方法。

完成这些更改后，`@trace_session("validation")`、`trace_session_phase("validation")` 和
`atrace_session_phase("validation")` 可以直接使用——上下文管理器将发出新事件，会话追踪器将为附加阶段序列化时间以及内置指标。

#### 绘制自定义阶段

在记录附加会话阶段后，使用 `areal/tools/plot_session_trace.py` 中的辅助函数来可视化它们。

1. **为新阶段着色** - 扩展 `SEGMENT_STYLES`，添加标签和颜色，以便生命周期时间线能明显呈现该跨度：

   ```python
   # areal/tools/plot_session_trace.py
   SEGMENT_STYLES["validation"] = {
        "label": "Validation",
        "color": "#14b8a6",
   }
   ```

   时间线渲染器自动拾取追踪负载中存在的每个阶段（它现在将默认的 generate/reward/toolcall 顺序与任何新键结合）。

1. **公开分布指标（可选）** - 如果 `SessionRecord` 发出一个派生的 `validation_s` 字段，将其添加到
   `DURATION_COLUMNS` 和 `HISTOGRAM_METRICS`，以便摘要图表显示每个阶段的直方图以及默认值。

1. **渲染报告** - 将会话 JSONL 文件指向绘图脚本并启用生命周期图表：

   ```bash
   python -m areal.tools.plot_session_trace \
     logs/**/session_tracer/sessions-r*.jsonl \
     --consumer-batch-size 512 \
     --enable-lifecycle
   ```

   这会在同一目录下生成 HTML 摘要，包括 `sessions-lifecycle-r*.html`，它在时间线上突出显示新阶段。

脚本从 `DEFAULT_PHASE_ORDER` 及其在追踪负载中发现的任何阶段名称推导绘图顺序，因此您在引入新阶段时只需扩展样式/指标字典。

## PyTorch Profiler 集成

`perf_tracer` 可以为目标步骤启动 PyTorch 分析器，以便您可以捕获内核和操作级数据，而无需在每次迭代中付出代价。

### 配置分析器触发步骤

- 将 `PerfTracerConfig.profile_steps` 设置为应运行分析器的全局步骤列表（0 索引）。示例：

  ```yaml
  perf_tracer:
      enabled: true
      profile_steps: [49, 99]
  ```

- 当任何 `trace_scope` 或 `atrace_scope` 带有 `args={"global_step": step}` 且 `step`
  匹配配置的条目之一时，`perf_tracer` 将检查是否应该为这个 `global_step` 触发分析。

- 将 `enable_profiler=True` 传递给 `trace_scope`/`atrace_scope`：作用域**仅**在*同时*设置了标志且
  `global_step` 匹配配置的 `profile_steps` 之一时进行分析。没有计划，手动请求会被忽略；没有标志，计划中的步骤保持休眠状态。

## 故障排除

**Q：追踪为空或缺少事件**

A：确保在退出之前运行 `perf_tracer.save(force=True)`。检查 `perf_tracer.configure()` 是否使用正确的 rank 调用。

**Q：会话追踪显示所有 `status: "pending"`**

A：生命周期事件（`mark_finalized`）未被记录。验证 `WorkflowExecutor` 是否正在调用
`trace_session_event()`，或者您的自定义 Workflow 是否实现了完整的生命周期。

**Q：Perfetto 无法打开我的追踪**

A：JSONL 格式需要转换。使用提供的转换器工具或手动包装在 JSON 数组中：

```bash
python -m areal.tools.perf_trace_converter traces.jsonl trace.json
```

**Q：某些会话的阶段显示 `end_ts` 为 `null`**

A：这发生在 `engine.agenerate()` 抛出异常并传播到 `arun_episode` 时。协调器然后调用
`trace_session_event("mark_finalized", task_id=task_id, ...)`，这将最终化该任务下的**所有**会话——包括那些阶段在执行中断裂的会话，留下
`end_ts: null`。

**解决方案**：远程推理引擎**不应从 `agenerate()` 抛出异常**。在内部处理错误并返回错误响应。

## 另请参阅

- {ref}`CLI 参考：PerfTracerConfig <section-perf-tracer>`
- [Workflow 自定义指南](../customization/agent.md)
- [Chrome Trace 事件格式](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [Perfetto UI](https://ui.perfetto.dev/)
