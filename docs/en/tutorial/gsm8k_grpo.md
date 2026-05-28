# Running GRPO on GSM8K Dataset

This guide walks you through how AReaL runs the GRPO algorithm on the GSM8K dataset.
We'll use the example training script
[`examples/math/gsm8k_rl.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)
and configuration file
[`examples/math/gsm8k_grpo.yaml`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml)
to explain the key concepts step by step.

## Overview: How AReaL Works

### Single-Controller Architecture

AReaL uses a **single-controller architecture** where the training script orchestrates
remote workers via RPC:

```
Controller Process (Your Script)
    │
    ├─> RolloutController
    │   ├─> Manages rollout workers (SGLang/vLLM)
    │   ├─> Submits prompts to inference workers
    │   ├─> Collects trajectories
    │   └─> Returns: RTensor (distributed batch)
    │
    └─> TrainController
        ├─> Manages training workers (FSDP/Megatron)
        ├─> Dispatches RTensor via data_parallel_dispatch()
        ├─> Workers compute forward/backward
        ├─> Merges results via data_parallel_merge()
        └─> Returns: loss, metrics
```

**Training Step Flow**:

1. **Rollout Phase**: Controller loads data and passes it to RolloutController, which
   schedules and routes rollout requests to rollout workers (GPUs).

   - Each rollout worker serves a complete model (may occupy multiple GPUs)
   - Returns: RTensor with shards stored on rollout workers (controller holds only
     metadata)

1. **Dispatch Phase**: TrainController distributes work via `data_parallel_dispatch()`

   - Uses FFD (First Fit Decreasing) to balance sequence lengths across workers
   - Workers fetch their assigned shards directly from rollout workers

1. **Training Phase**: Each training worker processes its shard independently

   - Supports 5D parallelism (data, tensor, pipeline, context, expert)

1. **Weight Sync**: Transfer updated weights to inference workers

   - Via NCCL (fast, GPU-to-GPU) or disk (fallback)

### Data Flow with RTensor

```
Rollout Workers (GPUs 0-3)              Controller              Training Workers (GPUs 4-7)
─────────────────────────────          ────────────            ─────────────────────────────
Worker 0: Generates 16 samples
          ├─> Shard 0 stored ────────────┐
Worker 1: Generates 16 samples           │
          ├─> Shard 1 stored ──────────┐ │
Worker 2: Generates 16 samples         │ │
          ├─> Shard 2 stored ────────┐ │ │
Worker 3: Generates 16 samples       │ │ │
          └─> Shard 3 stored ──────┐ │ │ │
                                   │ │ │ │
                                   │ │ │ │    RTensor metadata
                                   │ │ │ └─> Controller ─> data_parallel_dispatch()
                                   │ │ └───────────┼────────────┬────────────┐
                                   │ └─────────────┼────────────┼────────────┤
                                   └───────────────┼────────────┼────────────┤
                                                   │            │            │
                                                   ▼            ▼            ▼
                                               Worker 4:    Worker 5:    Worker 6:
                                               Fetch        Fetch        Fetch
                                               Shards 0,1   Shards 2     Shards 3
                                                   │            │            │
                                               ├─> Forward  ├─> Forward  ├─> Forward
                                               ├─> Backward ├─> Backward ├─> Backward
                                               └─> Grads    └─> Grads    └─> Grads
                                                            │
                                                     NCCL AllReduce
                                                            │
                                               Worker 4:    Worker 5:    Worker 6:
                                               Returns      Returns      Returns
                                               RTensor      RTensor      RTensor
                                                   │            │            │
                                                   └────────────┴────────────┘
                                                                │
                                                     data_parallel_merge()
                                                                │
                                                                ▼
                                                      Controller receives:
                                                      • loss (scalar)
                                                      • metrics (dict)
```

In the following sections, we'll walk through the code to explain each component in
detail.

## Launching the Experiment

AReaL supports launching experiments with different scheduler backends for different
environments. As shown in the [quickstart guide](../tutorial/quickstart.md), you can
launch experiments with:

```bash
# Local machine (using subprocesses)
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local

# Ray cluster
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=ray

# Slurm cluster
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=slurm
```

### How Single-Controller Mode Works

**Training Script**: Your experiment entry point (e.g., `examples/math/gsm8k_rl.py`)
that runs on the controller node.

**Controller Responsibilities**:

1. Controllers create worker processes (an HTTP or Ray server)
   `scheduler.create_workers()`
1. After workers are created, controllers create engines (e.g., `RemoteSGLangEngine`,
   `FSDPEngine`) via `scheduler.create_engine()`
1. Controllers dispatch work via RPC and coordinate via PyTorch distributed primitives

**Key Configuration**:

- `scheduler.type`: Determines which backend to use (`local`, `ray`, or `slurm`)
- Per-engine `backend` fields (e.g., `rollout.backend`, `actor.backend`): Determine GPU
  allocation and parallel strategies for each engine
- Schedulers automatically handle worker placement, resource allocation, and lifecycle
  management

### Configuration Files

Configuration files are YAML files that specify options from
[`areal/api/cli_args.py`](https://github.com/areal-project/AReaL/blob/main/areal/api/cli_args.py).
You can override settings via CLI:

```bash
# Example: change model and attention backend
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    actor.path=Qwen/Qwen3-1.7B \
    +sglang.attention_backend=triton
```

In your training script, parse the configuration:

```python
config, _ = load_expr_config(args, GRPOConfig)
config: GRPOConfig
```

See [CLI Reference](../cli_reference.md) for all available options.

## The Training Script: Entry Point

The training script
([`examples/math/gsm8k_rl.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py))
follows this pattern:

```python
def main(args):
    # 1. Load config (YAML + CLI overrides)
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 2. Prepare datasets (loaded on controller)
    train_dataset = get_custom_dataset(split="train", dataset_config=config.train_dataset, tokenizer=tokenizer)
    valid_dataset = get_custom_dataset(split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer)

    # 3. Define workflow configuration (imported on workers)
    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )

    # 4. Train with PPOTrainer
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )
```

**Key Points:**

- Datasets loaded on controller, then distributed to workers by controllers
- Workflows specified as import strings to enable dynamic instantiation on remote
  workers
- `PPOTrainer` handles all infrastructure (scheduler, controllers, workers)

See [CLI Reference](../cli_reference.md) for configuration options, and
[Customization: Dataset](../customization/dataset.md) for custom datasets.

## The PPOTrainer: Controller-Based Training

The
[`PPOTrainer`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
orchestrates distributed training by initializing the scheduler and creating controllers
for actors (policy/critic) and rollout workers.

### Controller Architecture

```
PPOTrainer (Controller Process)
    │
    ├── actor: PPOActorController (TrainController)
    │   ├── scheduler.create_workers() → Training workers
    │   ├── Remote engines: FSDPPPOActor instances
    │   └── APIs: compute_logp(), compute_advantages(), ppo_update()
    │
    ├── rollout: RolloutController
    │   ├── scheduler.create_engine() → Inference workers (SGLang/vLLM)
    │   ├── BatchTaskDispatcher → Async workflow execution
    │   └── API: prepare_batch() → Returns batch tensors
    │
    └── ref: PPOActorController (optional)
        └── Frozen reference model for KL penalty
```

**Key Pattern**: Engines use `as_controller(config, scheduler)` to wrap themselves in
controllers. The controller handles worker creation, RPC dispatch, and result merging.

## Rollout: Generating Training Data

### Workflow Specification

In `examples/math/gsm8k_rl.py`, workflows are specified as strings to enable dynamic
importing on remote workers:

```python
trainer.train(
    workflow="areal.workflow.rlvr.RLVRWorkflow",
    workflow_kwargs={
        "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
    },
)
```

### RLVRWorkflow: Single-Turn Reward Learning

The
[`RLVRWorkflow`](https://github.com/areal-project/AReaL/blob/main/areal/workflow/rlvr.py)
defines how prompts become training samples. Each trajectory goes through these steps:

1. **Tokenize input**: Apply chat template to messages
1. **Generate response**: Call inference engine (SGLang/vLLM)
1. **Compute reward**: Evaluate completion against ground truth
1. **Build training sample**: Construct tensor dict with:
   - `input_ids`: Full sequence (prompt + completion)
   - `loss_mask`: 0 for prompt tokens, 1 for completion tokens
   - `logprobs`: Log probabilities from generation
   - `versions`: Model version for each token (-1 for prompt)
   - `rewards`: Scalar reward

**GSM8K Reward**: Binary reward (1.0 for correct answer, 0.0 otherwise). See
[`gsm8k_reward_fn`](https://github.com/areal-project/AReaL/blob/main/areal/reward/gsm8k.py).

**NOTE:** This workflow adopts the low-level API of inference engines --- the
`agenerate` API. It is preferable if you want more fine-grained control over token IDs.
`agenerate` inputs token IDs to the inference server and produces output token IDs for
user's processing. We also provide high-level API for convenient agentic workflow
orchestration. We refer to the [agentic RL guide](../tutorial/agentic_rl.md).

### Asynchronous Rollout Collection

Rollout in AReaL is fully asynchronous with three levels of concurrency that enable
overlap between generation and training.

#### Three-Process Architecture

```
Controller Process              Worker Process (RPC Server)         GPU Process
──────────────────              ───────────────────────────         ───────────
RolloutController               Flask HTTP Server (CPU)             SGLang/vLLM
    │                               │                                   │
    └─> BatchTaskDispatcher     /call endpoint                      Inference
        (background thread)         │                               Engine
            │                       └─> Engine Thread                   │
            ├─ submit task 1            └─> RemoteInfEngine             │
            │  (HTTP POST)                  └─> submit() ──────────────>│
            │                                                        Generate
            ├─ submit task 2                                         tokens
            │  (HTTP POST)                                              │
            │                                                           │
            ├─ submit task 3              HTTP Callback  <──────────────┘
            │                             (trajectory)
            │                  ┌─────────────┘
            └─ collect  <──────┘

Meanwhile (on different GPUs)...
TrainController                 Training Worker
    │                               │
    └─> ppo_update(batch) ──────────> Forward/Backward

Key: Generation and training happen SIMULTANEOUSLY on different GPUs
```

#### Three Levels of Concurrency

**Level 1 - Controller Thread**:
[`BatchTaskDispatcher`](https://github.com/areal-project/AReaL/blob/main/areal/core/workflow_executor.py)
runs in a background thread, continuously submitting rollout requests to workers via
HTTP:

- Submits tasks round-robin to rollout workers
- Maintains 2+ batches of inflight requests to hide latency
- Non-blocking: returns task_id immediately

As such, **rollout and training happen simultaneously** in AReaL, even though the code
looks like a synchronous orchestration.

**Level 2 - Worker RPC Server**: Each rollout worker runs a Flask HTTP server
([`rpc_server.py`](https://github.com/areal-project/AReaL/blob/main/areal/infra/rpc/rpc_server.py))
on **CPU**:

- Accepts concurrent HTTP requests (multi-threaded Flask)
- **Engine thread**: Processes engine operations serially (NCCL compatibility)
- Routes requests to `RemoteInfEngine` which queues work to SGLang/vLLM

**Level 3 - GPU Subprocess**: SGLang/vLLM runs as a **separate subprocess on GPU**:

- Launched via `backend.launch_server()` (separate from RPC server)
- Maintains its own request queue
- Processes multiple concurrent generations with continuous batching
- Sends HTTP callbacks when trajectories complete

#### Request Flow

```python
# 1. Controller calls prepare_batch
batch = rollout.prepare_batch(
    dataloader,
    workflow="areal.workflow.rlvr.RLVRWorkflow",
    workflow_kwargs=workflow_kwargs,
)

# 2. RolloutController delegates to BatchTaskDispatcher
# Background thread submits tasks:
for data in dataloader:
    task = _RemoteRolloutTaskInput(data, workflow, workflow_kwargs, task_id)
    dispatcher.submit_task_input(task)  # Non-blocking HTTP POST

# 3. Worker RPC server receives HTTP POST /call (method="submit")
# Engine thread executes:
workflow_instance = import_from_string(workflow)(**workflow_kwargs)
task_id = workflow_executor.submit(data, workflow_instance)
# Returns immediately (non-blocking)

# 4. WorkflowExecutor (on worker) runs in background:
result = await workflow_instance.arun_episode(engine, data)
# Sends HTTP callback to controller with trajectory

# 5. Controller collects results
# BatchTaskDispatcher waits for batch_size accepted trajectories
results = dispatcher.wait_results(batch_size)
return concat_padded_tensors(results)  # Shape: [batch_size, seq_len]
```

**Staleness Control**:
[`StalenessManager`](https://github.com/areal-project/AReaL/blob/main/areal/infra/staleness_manager.py)
limits concurrent inflight requests:

- `max_concurrent_rollouts`: Maximum inflight trajectories
- `max_head_offpolicyness`: Reject samples generated with weights too old
- Version tracking: Each token tagged with model version used during generation

## Training: Controller-Worker Pattern

Training follows a standard controller-worker pattern. The controller dispatches
algorithm operations to training workers via RPC, workers process their data shards, and
results are merged back.

### TrainController: Dispatch Mechanism

[`TrainController`](https://github.com/areal-project/AReaL/blob/main/areal/infra/controller/train_controller.py)
provides the core RPC dispatch:

1. `_dispatch_inputs()`: Splits batches using FFD load balancing across workers
1. RPC calls: Each worker receives its shard, processes it, returns results
1. `_merge_results()`: Reconstructs results from data-parallel workers

**Data Flow with RTensor:**

```
Controller                  Worker 0                  Worker 1
    │                           │                         │
    ├─ RTensor (metadata) ──────┼─────────────────────────┤
    │  • Shards 0,1,2,3         │                         │
    │                           │                         │
    ├─ dispatch() ────────────> │                         │
    │  • Worker 0: Shards 0,1   │                         │
    │  • Worker 1: Shards 2,3   │                         │
    │                           │                         │
    │                           ├─> Fetch Shards 0,1      │
    │                           │   from rollout workers  │
    │                           │                         ├─> Fetch Shards 2,3
    │                           │                         │   from rollout workers
    │                           │                         │
    │                           ├─> compute_logp()        ├─> compute_logp()
    │                           │                         │
    │                           ├─> RTensor (result)      ├─> RTensor (result)
    │<─ merge() ────────────────┴─────────────────────────┘
    │  • Reconstruct ordering
    │  • Return unified RTensor
    └─> batch["logp"] = result
```

**Key Design**: Controller only handles metadata (RTensor). Workers fetch actual tensors
directly from rollout workers, avoiding controller memory overhead.

### Training Workers: Algorithm Implementation

On each training worker,
[`FSDPPPOActor`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/ppo/actor.py)
implements the GRPO/PPO algorithm:

**Algorithm Methods:**

- `compute_logp(batch)`: Forward pass through model to compute log probabilities
- `compute_advantages(batch)`: Apply reward/advantage normalization (group or batch
  level)
- `ppo_update(batch)`: Policy update with mini-batch training and gradient accumulation
  - Splits batch into mini-batches
  - Computes PPO loss (clipped surrogate objective + optional KL penalty)
  - Performs backward pass and optimizer step

**Parallelism**: Each engine's `backend` field determines its GPU allocation:

```
rollout.backend=sglang:d4 actor.backend=fsdp:d4, n_gpus=8

Rollout Workers:     Training Workers:
GPU 0: SGLang        GPU 4: FSDP rank 0  ─┐
GPU 1: SGLang        GPU 5: FSDP rank 1   ├─ Data Parallel
GPU 2: SGLang        GPU 6: FSDP rank 2   │  (DP size = 4)
GPU 3: SGLang        GPU 7: FSDP rank 3  ─┘
                           │
                     NCCL AllReduce for gradients
```

Each worker processes its shard, then synchronizes gradients via NCCL.

### The Training Loop

The `trainer.train()` method orchestrates the complete loop. See
[`PPOTrainer.train()`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
for the full implementation:

```python
for global_step in range(start_step, max_steps):
    # 1. Rollout
    rollout_batch = self.actor.prepare_batch(train_dataloader, workflow, workflow_kwargs)

    # 2. Compute log-probs and advantages
    if config.actor.should_compute_prox_logp():
        rollout_batch["prox_logp"] = self.actor.compute_logp(rollout_batch)
    if self.ref:
        rollout_batch["ref_logp"] = self.ref.compute_logp(rollout_batch)
    adv_batch = self.actor.compute_advantages(rollout_batch)

    # 3. PPO update
    self.actor.ppo_update(adv_batch)
    self.actor.step_lr_scheduler()

    # 4. Weight sync
    self.rollout.pause()
    self.actor.update_weights(weight_update_meta)
    self.actor.set_version(global_step + 1)
    self.rollout.set_version(global_step + 1)
    self.rollout.resume()
```

All algorithm operations are controller method calls that dispatch to remote workers.

## Weight Synchronization

After each training step, updated weights must be synced to inference workers. AReaL
supports two transfer methods:

### Transfer Methods

**NCCL-based transfer** (Recommended):

- Direct GPU-to-GPU communication based on NCCL broadcast
- Faster but uses more GPU memory
- Requires non-overlapped training and inference GPUs on the same communication backend

**Disk-based transfer**:

- Saves to shared storage, then loads on inference servers
- Use when NCCL is unavailable or machines don't share a backend

### Weight Update Process

The weight sync process in `PPOTrainer.train()` follows this pattern:

1. Pause rollout servers to interrupt all inflight generations back to the rollout
   client (e.g., `RemoteSGLangEngine`)
1. Transfer weights via configured method (NCCL or disk)
1. Update version tracking for staleness management
1. Resume rollout with updated weights with re-computed KV cache

See
[`PPOTrainer.train()`](https://github.com/areal-project/AReaL/blob/main/areal/trainer/rl_trainer.py)
lines 861-874 for the full implementation.

## Monitoring and Utilities

AReaL provides utilities managed by `PPOTrainer` for checkpointing, evaluation, and
metrics tracking. These are automatically orchestrated during training.

### Checkpointing

AReaL provides two checkpointing mechanisms:

| Component                                                                                   | Purpose                          | Format        | Configuration    |
| ------------------------------------------------------------------------------------------- | -------------------------------- | ------------- | ---------------- |
| [`Saver`](https://github.com/areal-project/AReaL/blob/main/areal/utils/saver.py)            | Export for evaluation/deployment | HuggingFace   | `config.saver`   |
| [`RecoverHandler`](https://github.com/areal-project/AReaL/blob/main/areal/utils/recover.py) | Resume after failures            | DCP (sharded) | `config.recover` |

**Saver** creates HuggingFace-compatible checkpoints that can be loaded with
`transformers` or published to HuggingFace Hub. Each save creates a new directory.

**RecoverHandler** saves complete training state (model, optimizer, dataloader, RNG) for
fault tolerance. Checkpoints are backend-specific and require the same parallelism
configuration to load. Each save overwrites the previous checkpoint.

Both are called automatically during `trainer.train()`. For details, see the
[Checkpointing Reference](../reference/checkpointing.md).

### Evaluation

The
[`Evaluator`](https://github.com/areal-project/AReaL/blob/main/areal/utils/evaluator.py)
runs periodic evaluations on validation sets. Configure via `config.evaluation`. Called
automatically in `trainer.train()`.

### Metrics Tracking

AReaL uses a two-component metrics system:

**`stats_tracker`**
([source](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_tracker.py)):
Collects statistics with two paradigms optimized for different use cases:

- **Streaming metrics** for rollout workers: Each workflow logs scalars individually
  (e.g., `reward`), which are aggregated across workers by the controller
- **Batch metrics** for training: Tensor statistics with boolean masks are logged per
  batch, then all-reduced across data-parallel ranks

```python
# Rollout metrics (streaming) - in workflows
stats_tracker.get("rollout").scalar(reward=0.8, num_turns=3)

# Training metrics (batch) - in PPO actor
stats_tracker.denominator(n_valid_tokens=loss_mask.bool())
stats_tracker.stat(advantages=tensor, denominator="n_valid_tokens")
```

**`StatsLogger`**
([source](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_logger.py)):
Sends aggregated metrics to logging backends (Weights & Biases, SwanLab, TensorBoard)
from rank 0. At each training step, `PPOTrainer` collects metrics from all components
and commits them:

```python
# areal/trainer/rl_trainer.py
stats = self.actor.export_stats()         # Training metrics
stats.update(self.rollout.export_stats()) # Rollout metrics
self.stats_logger.commit(epoch, step, global_step, stats)  # → wandb/tensorboard
```

For the complete API reference, see the
[Metrics Tracking Reference](../reference/metrics_tracking.md).

## Next Steps

Now that you understand the basics, explore these advanced topics:

**Tutorials**:

- [Evaluation](../tutorial/eval.md) - Evaluate your trained model
- [Training Large MoE Models](../tutorial/megatron.md) - Scale to massive models with
  Megatron integration
- [Agentic RL](../tutorial/agentic_rl.md) - Build agents that use tools and any agentic
  frameworks

**Customization Guides**:

- [Custom Datasets](../customization/dataset.md) - Use your own data sources
- [Custom Workflows](../customization/agent.md) - Build agentic/RLVR workflows with
  custom reward functions
