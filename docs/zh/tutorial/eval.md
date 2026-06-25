# 评估

AReaL 支持使用与训练相同的控制器基础设施进行分布式推理。这允许您利用现有的工作流和调度器在多个 GPU 和节点上扩展评估。

**注意：** AReaL 为您训练好的模型提供分布式推理，而不是包含数据集获取和指标计算的完整评估管道。您可以直接将 AReaL
检查点与第三方评估框架配合使用——无需转换，因为 AReaL 保存的是 HuggingFace 兼容的检查点。

## 快速开始

在 GSM8K 上运行评估：

```bash
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    actor.path=/path/to/checkpoint
```

对于分布式评估：

```bash
# 使用 Ray（3 个节点，12 个 GPU）
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=ray \
    rollout.backend=sglang:d12p1t1 \
    cluster.n_nodes=3

# 使用 Slurm（12 个节点，96 个 GPU）
python3 examples/math/gsm8k_eval.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=slurm \
    rollout.backend=sglang:d96p1t1 \
    cluster.n_nodes=12
```

## 评估指标

为您的任务选择合适的数据集和指标，然后将评估逻辑集成到工作流中。请参阅[智能体 RL 指南](./agentic_rl.md)了解详情。

以下是使用智能体数学评估器的示例（评估代码独立于 AReaL）：

```python
from agents import Agent, OpenAIProvider, RunConfig, SQLiteSession, function_tool
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI


@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@function_tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def math_reward_fn(completions: str, answer: str) -> float:
    return float(verify(parse(completions), parse(answer)))


class MathAgent:
    async def run(self, data, **extra_kwargs):
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)

        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",
            tracing_disabled=True,
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer math questions using the calculator tools.",
            tools=[add, multiply],
        )
        result = await OpenAIRunner.run(
            agent,
            input=data["messages"][-1]["content"],
            session=SQLiteSession("math"),
            run_config=run_config,
        )
        return math_reward_fn(result.final_output, data["answer"])
```

## 架构

评估使用单一控制器架构，没有训练 worker：

```
Controller Process
    │
    └─> Inference Engine Controller (SGLang/vLLM)
        ├─> Scheduler creates inference workers
        ├─> Submits evaluation tasks with workflow
        └─> Collects results and computes metrics
```

控制器在 CPU 进程中协调评估，而推理 worker 在 GPU 上运行。

## 实现

有关完整示例，请参阅
[`examples/math/gsm8k_eval.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_eval.py)。关键模式如下：

```python
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler

# 加载配置并解析 rollout 后端
config, _ = load_expr_config(args, GRPOConfig)
rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")

# 根据配置初始化调度器
if config.scheduler.type == "local":
    scheduler = LocalScheduler(exp_config=config)
elif config.scheduler.type == "ray":
    scheduler = RayScheduler(exp_config=config)
elif config.scheduler.type == "slurm":
    scheduler = SlurmScheduler(exp_config=config)

# 选择推理引擎并构建服务器参数
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

# 创建控制器并初始化
eval_rollout = engine_cls.as_controller(config.rollout, scheduler)
eval_rollout.initialize(
    role="eval-rollout",
    server_args=server_args,
)

# 定义工作流及其配置
workflow = "areal.workflow.rlvr.RLVRWorkflow"
workflow_kwargs = dict(
    reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
    enable_thinking=False,
)

# 提交评估任务
cnt = 0
for data in valid_dataloader:
    for item in data:
        eval_rollout.submit(
            item,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=config.gconfig.n_samples,
        )
        cnt += 1

# 等待完成并收集结果
eval_rollout.wait(cnt, timeout=None)
eval_stats = eval_rollout.export_stats()
```

这遵循与训练相同的控制器模式，但没有训练组件。

## 配置

评估重用了与训练相同的配置结构。您可以直接将现有的训练配置与评估脚本一起使用。

```yaml
experiment_name: gsm8k-eval
trial_name: eval0
seed: 1

scheduler:
  type: local  # 或 'ray', 'slurm'

rollout:
  backend: "sglang:d4p1t1"  # 仅推理分配
  max_concurrent_rollouts: 256
  # max_head_offpolicyness 在内部设置为 1e12 用于评估

gconfig:
  n_samples: 8
  temperature: 1.0
  max_new_tokens: 1024

actor:
  path: Qwen/Qwen2.5-1.5B-Instruct
  dtype: bfloat16
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cmd: python3 -m areal.infra.rpc.rpc_server

valid_dataset:
  name: gsm8k
  split: test
  batch_size: 32
```

## 记录结果

使用 `tabulate_stats` 来格式化评估指标：

```python
from areal.utils.printing import tabulate_stats

eval_stats = eval_rollout.export_stats()
logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")
```

## 自定义工作流

重用训练工作流或创建自定义工作流。请参阅[智能体 RL 教程](../tutorial/agentic_rl.md)和[自定义：Rollout 工作流](../customization/agent.md)获取完整指南。

## 下一步

- {ref}`使用 Ray 或 Slurm 的分布式实验 <distributed-experiments-with-ray-or-slurm>`
- [自定义：工作流](../customization/agent.md)
- [智能体 RL 教程](../tutorial/agentic_rl.md)
- [大型 MoE 训练](../tutorial/megatron.md)
