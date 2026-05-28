# Debugging Guide

This guide covers debugging AReaL training applications, including:

- Debugging your agent workflow (i.e., rollout) with a persistent inference server
- Comparing rollout results between Transformers and inference engines
- Diagnosing training hangs and deadlocks with `py-spy`

## Debugging agent workflows with a Persistent Inference Server

In AReaL, any class with the method signature
`async def run(self, data, **extra_kwargs)` is recognized as an agent workflow. This
method can use arbitrary agentic frameworks internally and must return a scalar or a
dict of rewards assigning the credit for each LLM interaction. See
[Agentic RL guide](../tutorial/agentic_rl.md) for more details.

You can use the official OpenAI/Anthropic API or launch a **standalone, persistent
inference server** for your agent's generation logic, enabling repeated testing without
system restarts.

**Benefits:**

- **Lightweight** - Your debug program runs on CPU while inference runs on the
  provider's GPU
- **IDE-friendly** - Works seamlessly with VS Code's Python debugger and other IDEs
- **Fast iterations** - No server restarts needed between debugging sessions

### 1. (Optional) Launch the Standalone Inference Server

**NOTE**: You can skip this step if you want to use official model providers for
debugging

This example uses `sglang`, but any framework that exposes OpenAI/Anthropic HTTP
endpoints works (e.g., `vllm`).

Start your server following the [official documentation](https://docs.sglang.io/):

```bash
nohup python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 8080 --log-level warning > llm_server.log 2>&1 &
```

Once it's running, you'll find the server address in the log:

```
[2026-02-06 15:38:30] INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 2. Debug your agent with an individual run

Run your agent with proper base URL and API key:

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

Test your code with random samples from the dataset. If it runs without errors, your
agent logic is correct.

### 3. Debug your agent with many concurrent runs

RL training typically requires generating large batches, so you should verify that your
agent code (**especially the internal agentic framework**) can handle high concurrency.
Some frameworks target single-threaded scenarios and may not scale well for RL training.

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

### 4. Integration Test with AReaL

Once all previous steps pass, you can integrate the workflow into AReaL.

Place your agent in an importable path, e.g., `my_agent.MyAgent`, then initialize a
rollout controller in AReaL to do batched rollout:

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

Run the script with:

```bash
python3 script.py --config xxx.yaml scheduler.type=local
```

This essentially follows the same procedure as [evaluation](../tutorial/eval.md).

**IMPORTANT**:

1. Use the same configuration file as training; irrelevant fields are ignored.
1. Ensure `max_head_offpolicyness` and `max_concurrent_rollouts` are large enough,
   otherwise the rollout process will block indefinitely due to staleness control.

If step 4 passes, your code is ready for AReaL. You can now pass it to the trainer and
start training.

## Rollout Consistency

Comparing rollout results between `transformers` and your inference engine helps verify
consistency and correctness. While most models produce nearly identical results, some
may exhibit significant differences due to the extensive optimizations that inference
backends (e.g., `sglang`, `vllm`) apply to accelerate the forward pass.

If you suspect discrepancies, or if you're working with models lacking first-class
support in Transformers or SGLang, compare outputs against a dataset using a simple
validation script. See `examples/docs/debug/cmp_rollout.py` for a complete example
comparing rollout results for `google/gemma-3-4b-it` on the
`BUAADreamer/clevr_count_70k` dataset.

## Debugging Training Hangs and Deadlocks

Distributed training can hang or deadlock when ranks get out of sync. This section
covers how to diagnose these issues.

### Symptoms

A hang or deadlock typically looks like one of the following:

- **Training stops making progress** — logs stop updating, no new training steps, but
  processes remain alive with high CPU usage.
- **Training exits with no error** — the job finishes (sometimes even printing "Training
  completes!") but completed 0 actual training steps. The processes may hang
  indefinitely during cleanup.
- **Some ranks finish, others hang** — `nvidia-smi` shows some GPUs idle while others
  are still at 100% utilization.

These symptoms usually share a common root cause: **an exception or early exit on some
ranks causes the remaining ranks to wait forever on a collective operation** (e.g.,
`all_reduce`, `send`/`recv`, or `destroy_process_group`).

### Common Causes

| Cause                           | What happens                                                                                                                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exception on partial ranks**  | One side of a PP/TP group hits an error and exits, while the other side waits for a P2P or collective op that never arrives. The exception may be swallowed by cleanup code (`__exit__` → `destroy_process_group()` hang). |
| **Mismatched collective calls** | A code path calls `all_reduce` on some ranks but not others (e.g., due to a conditional branch that differs across ranks).                                                                                                 |
| **Shape mismatch in PP**        | Pipeline parallel stages expect to exchange tensors of specific shapes. If one stage produces unexpected shapes, `recv` blocks forever.                                                                                    |
| **NCCL timeout**                | Network issues or slow ranks cause NCCL operations to exceed the timeout, but the default timeout may be very long (30 minutes).                                                                                           |
| **Deadlock in initialization**  | Model loading or compilation takes different amounts of time across ranks, and a collective is called before all ranks are ready.                                                                                          |

### Step 1: Confirm the Hang

First, verify that training is actually hung (not just slow):

```bash
# Check if training steps are advancing
tail -f /path/to/training.log

# Check GPU utilization — hung ranks often show 0% GPU, high CPU
nvidia-smi

# List the training processes
ps aux | grep 'python.*areal' | grep -v grep
```

### Step 2: Dump Call Stacks with `py-spy`

[py-spy](https://github.com/benfred/py-spy) is the most effective tool for diagnosing
hangs. It attaches to a running Python process and dumps the call stack without
interrupting execution.

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

### Step 3: Read the Call Stacks

The call stacks tell you exactly where each rank is blocked. Look for these patterns:

**Pattern A: Cleanup deadlock** — Some ranks finished (hit an error or completed early)
and are stuck in `destroy_process_group`, while others are still in the training loop
waiting for communication:

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

**Pattern B: Collective mismatch** — All ranks are inside the training loop, but waiting
on different collective operations:

```
# Rank 0
    all_reduce (torch/distributed/distributed_c10d.py)
    forward (some_module.py:123)

# Rank 1
    all_reduce (torch/distributed/distributed_c10d.py)
    backward (some_module.py:456)       ← different code path!
```

**Pattern C: NCCL timeout** — All ranks are in the same collective call, suggesting a
network or performance issue rather than a code bug:

```
# All ranks show the same stack:
    all_reduce (torch/distributed/distributed_c10d.py)
    forward (my_model.py:100)           ← same location on all ranks
```

### Step 4: Environment Variables for More Detail

Set these environment variables **before** launching training to get more information
when hangs occur:

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

### Tips

- **Exception swallowed by cleanup**: If py-spy shows some ranks in
  `destroy_process_group` and others still in the training loop, the root cause is an
  exception on the exiting ranks.
- **Reproduce with fewer GPUs**: If possible, reproduce with the minimum number of GPUs
  (e.g., PP=2 with 2 GPUs). This makes the call stacks easier to read.
- **Check all ranks**: Always dump stacks for **all** worker processes, not just one.
  Hangs are fundamentally about rank divergence — you need to compare stacks across
  ranks to see who is waiting for whom.
