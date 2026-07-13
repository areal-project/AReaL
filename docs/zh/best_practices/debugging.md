# 调试指南

本指南涵盖 AReaL 训练应用的调试，包括：

- 使用持久化推理服务器调试您的 Agent Workflow（即 rollout）
- 比较 Transformers 与推理引擎之间的 rollout 结果
- 使用 `py-spy` 诊断训练挂起和死锁问题

## 使用持久化推理服务器调试 Agent Workflow

在 AReaL 中，任何具有方法签名 `async def run(self, data, **extra_kwargs)` 的类都被识别为 Agent
Workflow。该方法可以在内部使用任意的 Agentic 框架，并且必须返回标量或奖励字典，为每次 LLM 交互分配信用。更多详细信息请参阅
[Agentic RL 指南](../tutorial/agentic_rl.md)。

您可以使用官方 OpenAI/Anthropic API 或启动**独立的持久化推理服务器**来运行 Agent 的生成逻辑，从而无需重启系统即可进行重复测试。

**优势：**

- **轻量级** - 您的调试程序在 CPU 上运行，而推理在提供商的 GPU 上进行
- **IDE 友好** - 与 VS Code 的 Python 调试器和其他 IDE 无缝协作
- **快速迭代** - 调试会话之间无需重启服务器

### 1. （可选）启动独立推理服务器

**注意**：如果您想使用官方模型提供商进行调试，可以跳过此步骤

此示例使用 `sglang`，但任何暴露 OpenAI/Anthropic HTTP 端点的框架都可以（例如 `vllm`）。

按照[官方文档](https://docs.sglang.io/)启动您的服务器：

```bash
nohup python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 8080 --log-level warning > llm_server.log 2>&1 &
```

运行后，您可以在日志中找到服务器地址：

```
[2026-02-06 15:38:30] INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 2. 使用单次运行调试您的 Agent

使用正确的 base URL 和 API key 运行您的 Agent：

```python
import asyncio

from openai import AsyncOpenAI


class MyAgent:
    async def run(self, data, **extra_kwargs):
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        async with AsyncOpenAI(base_url=base_url, api_key=api_key) as client:
            comp = await client.chat.completions.create(
                model="qwen/qwen2.5-0.5b-instruct",
                messages=data["messages"],
                temperature=0,
                max_tokens=64,
            )
        return 1.0  # random reward


data = dict(messages=[{"role": "user", "content": "List 3 countries and their capitals."}])

# If you use a local inference server
port = 8080
asyncio.run(MyAgent().run(data, base_url=f"http://127.0.0.1:{port}/v1", api_key="None"))

# If you use the official model provider
asyncio.run(MyAgent().run(data, base_url="https://api.openai.com/v1", api_key="YOUR_API_KEY"))
```

使用数据集中的随机样本测试您的代码。如果它能无错误运行，则您的 Agent 逻辑是正确的。

### 3. 使用多个并发运行调试您的 Agent

RL 训练通常需要生成大批量，因此您应该验证您的 Agent 代码（尤其是内部使用的 Agentic 框架）能够处理高并发性。某些框架针对单线程场景设计，可能无法很好地扩展到
RL 训练。

```python
# Example GRPO configuration
global_bs = 256
group_size = 8

# Example allocation
rollout_dp_size = 4

local_bs = global_bs // rollout_dp_size
port = 8080


async def run_agent(data):
    return await MyAgent().run(data, base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


async def grouped_rollout(data):
    return await asyncio.gather(*[run_agent(data) for _ in range(group_size)])


async def batched_rollout(batch):
    assert len(batch) == local_bs
    return await asyncio.gather(*[grouped_rollout(data) for data in batch])


# batch should be a list of data dicts with length == local_bs
batch = [data] * local_bs
asyncio.run(batched_rollout(batch))
```

### 4. 与 AReaL 的集成测试

完成所有前序步骤后，您可以将 Workflow 集成到 AReaL 中。

将您的 Agent 放在可导入的路径中，例如 `my_agent.MyAgent`，然后在 AReaL 中初始化 rollout 控制器以进行批量 rollout：

```python
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
import sys

# Load config and parse rollout backend
config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
rollout_alloc = ModelAllocation.from_str(config.rollout.backend)

# Initialize scheduler based on config
if config.scheduler.type == "local":
    scheduler = LocalScheduler(exp_config=config)
elif config.scheduler.type == "ray":
    scheduler = RayScheduler(exp_config=config)
elif config.scheduler.type == "slurm":
    scheduler = SlurmScheduler(exp_config=config)

# Select inference engine and build server args
if rollout_alloc.backend == "sglang":
    engine_cls = RemoteSGLangEngine
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=rollout_alloc.parallel.tp_size,
        base_gpu_id=0,
    )
elif rollout_alloc.backend == "vllm":
    engine_cls = RemotevLLMEngine
    server_args = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=rollout_alloc.parallel.tp_size,
        pp_size=rollout_alloc.parallel.pp_size,
    )

# Create controller and initialize
eval_rollout = engine_cls.as_controller(config.rollout, scheduler)
eval_rollout.initialize(
    role="eval-rollout",
    server_args=server_args,
)

# Define workflow and its configuration
workflow = "areal.workflow.rlvr.RLVRWorkflow"
workflow_kwargs = dict(
    reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
    enable_thinking=False,
)

batch = eval_rollout.rollout_batch(
    batch,
    workflow=workflow,
    workflow_kwargs=workflow_kwargs,
    group_size=config.gconfig.n_samples,
)
```

使用以下命令运行脚本：

```bash
python3 script.py --config xxx.yaml scheduler.type=local
```

这基本上遵循与[评估](../tutorial/eval.md)相同的过程。

**重要提示**：

1. 使用与训练相同的配置文件；不相关的字段会被忽略。
1. 确保 `max_head_offpolicyness` 和 `max_concurrent_rollouts` 足够大，否则 rollout
   进程将因过期控制而无限期阻塞。

如果步骤 4 通过，您的代码已准备好用于 AReaL。您现在可以将其传递给训练器并开始训练。

## Rollout 一致性

比较 `transformers` 与您的推理引擎之间的 rollout 结果有助于验证一致性和正确性。虽然大多数模型产生的结果几乎相同，但某些模型可能由于推理后端（如
`sglang`、`vllm`）为加速前向传播而进行的大量优化而表现出显著差异。

如果您怀疑存在差异，或者如果您使用的模型在 Transformers 或 SGLang 中缺乏一流支持，请使用简单的验证脚本对照数据集比较输出。请参阅
`examples/docs/debug/cmp_rollout.py` 以获取完整的示例，比较 `google/gemma-3-4b-it` 在
`BUAADreamer/clevr_count_70k` 数据集上的 rollout 结果。

## 调试训练挂起和死锁

分布式训练可能会在 ranks 不同步时挂起或死锁。本节介绍如何诊断这些问题。

### 症状

挂起或死锁通常表现为以下情况之一：

- **训练停止进展** - 日志停止更新，没有新的训练步骤，但进程仍保持活跃且 CPU 使用率很高。
- **训练无错误退出** - 任务完成（有时甚至打印 "Training completes!"），但实际完成了 0 个训练步骤。进程在清理过程中可能会无限期挂起。
- **某些 ranks 完成，其他 ranks 挂起** - `nvidia-smi` 显示某些 GPU 空闲，而其他 GPU 仍处于 100% 利用率。

这些症状通常有一个共同的根本原因：**某些 ranks 上的异常或提前退出导致剩余 ranks 永远等待集合操作**（例如 `all_reduce`、`send`/`recv`
或 `destroy_process_group`）。

### 常见原因

| 原因                    | 发生了什么                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **部分 ranks 上的异常** | PP/TP 组的一方遇到错误并退出，而另一方等待永远不会到达的 P2P 或集合操作。异常可能被清理代码吞掉（`__exit__` → `destroy_process_group()` 挂起）。 |
| **集合调用不匹配**      | 代码路径在某些 ranks 上调用 `all_reduce` 但在其他 ranks 上不调用（例如由于跨 ranks 不同的条件分支）。                                            |
| **PP 中的形状不匹配**   | 流水线并行阶段期望交换特定形状的张量。如果某个阶段产生意外的形状，`recv` 将永远阻塞。                                                            |
| **NCCL 超时**           | 网络问题或慢 ranks 导致 NCCL 操作超过超时，但默认超时可能很长（30 分钟）。                                                                       |
| **初始化中的死锁**      | 模型加载或编译在不同 ranks 上花费不同时间，而在所有 ranks 准备好之前就调用了集合。                                                               |

### 步骤 1：确认挂起

首先，验证训练确实挂起了（不仅仅是慢）：

```bash
# Check if training steps are advancing
tail -f /path/to/training.log

# Check GPU utilization — hung ranks often show 0% GPU, high CPU
nvidia-smi

# List the training processes
ps aux | grep 'python.*areal' | grep -v grep
```

### 步骤 2：使用 `py-spy` 转储调用栈

[py-spy](https://github.com/benfred/py-spy) 是诊断挂起最有效的工具。它附加到正在运行的 Python
进程并转储调用栈而不会中断执行。

```bash
# Install py-spy (if not already installed)
pip install py-spy

# Dump call stack for a single process
py-spy dump --pid <PID>

# Dump all training worker processes at once
for pid in $(ps aux | grep 'python.*areal' | grep -v grep | awk '{print $2}'); do
    echo "========== PID $pid =========="
    py-spy dump --pid $pid
done
```

### 步骤 3：读取调用栈

调用栈会准确告诉您每个 rank 被阻塞的位置。寻找以下模式：

**模式 A：清理死锁** - 某些 ranks 完成（遇到错误或提前完成）并卡在 `destroy_process_group` 中，而其他 ranks
仍在训练循环中等待通信：

```
# Ranks that exited (e.g., PP Stage 0 hit an exception)
Thread: "MainThread"
    destroy_process_group (torch/distributed/distributed_c10d.py)
    destroy (archon_engine.py)
    close (sft_trainer.py)              ← stuck in cleanup
    __exit__ (sft_trainer.py)

# Ranks still running (e.g., PP Stage 1 waiting for data)
Thread: "MainThread"
    recv_object_list (torch/distributed/distributed_c10d.py)
    _shape_inference (torch/distributed/pipelining/stage.py)
    step (torch/distributed/pipelining/schedules.py)
    _run_train (archon_runner.py)       ← waiting for the other stage
```

**模式 B：集合不匹配** - 所有 ranks 都在训练循环内，但等待不同的集合操作：

```
# Rank 0
    all_reduce (torch/distributed/distributed_c10d.py)
    forward (some_module.py:123)

# Rank 1
    all_reduce (torch/distributed/distributed_c10d.py)
    backward (some_module.py:456)       ← different code path!
```

**模式 C：NCCL 超时** - 所有 ranks 都在相同的集合调用中，表明这是网络或性能问题而非代码错误：

```
# All ranks show the same stack:
    all_reduce (torch/distributed/distributed_c10d.py)
    forward (my_model.py:100)           ← same location on all ranks
```

### 步骤 4：用于获取更多详细信息的环境变量

在启动训练**之前**设置这些环境变量，以便在发生挂起时获取更多信息：

```bash
# NCCL debug logging — shows collective operations as they happen
export NCCL_DEBUG=INFO

# PyTorch distributed debug — logs every collective call with ranks and shapes
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Reduce NCCL timeout so hangs fail faster (default is 1800s = 30 min)
export NCCL_TIMEOUT=300  # 5 minutes

# CUDA sync mode — makes errors appear at the correct location
# WARNING: significant performance impact, use only for debugging
export CUDA_LAUNCH_BLOCKING=1
```

### 提示

- **异常被清理吞掉**：如果 py-spy 显示某些 ranks 在 `destroy_process_group` 中，而其他 ranks 仍在训练循环中，根本原因是退出
  ranks 上的异常。
- **使用更少的 GPU 重现**：如果可能，使用最少的 GPU 数量（例如 PP=2，2 个 GPU）重现。这会使调用栈更容易阅读。
- **检查所有 ranks**：始终转储**所有**工作进程的堆栈，而不仅仅是一个。挂起从根本上说是关于 rank 发散——您需要比较跨 ranks 的堆栈以了解谁在等待谁。
