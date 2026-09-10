# Metrics Tracking

AReaL provides a unified metrics tracking system that handles statistics collection
across distributed training and rollout workers. The system supports two distinct
paradigms optimized for their respective use cases: **streaming metrics** for
asynchronous rollout workflows and **batch metrics** for synchronous training updates.

## Core Components

The metrics system is built around `areal.utils.stats_tracker`, which provides:

- **Named trackers**: Isolated metric namespaces for different components
- **Hierarchical scoping**: Organize metrics into logical groups
- **Distributed aggregation**: Automatic reduction across workers
- **Multiple reduce types**: Support for averages, sums, min/max, and scalars

```python
from areal.utils import stats_tracker

# Default tracker (training metrics)
stats_tracker.scalar(learning_rate=0.001)

# Named tracker (rollout metrics)
stats_tracker.get("rollout").scalar(reward=0.5)
```

## Two Logging Paradigms

### Streaming Metrics (Rollout Workers)

Rollout workers execute workflows asynchronously, with each workflow logging metrics
independently. This streaming approach handles variable completion times naturally.

**Characteristics:**

- Each workflow logs scalars individually as they complete
- Metrics accumulate in a list within the worker process
- No synchronization between workers during logging
- Aggregation happens at export time via the controller

**Example from `RLVRWorkflow`:**

```python
# areal/workflow/rlvr.py
async def _collect_samples(self, engine, req, prompt_str, task_data):
    resp = await engine.agenerate(req)
    reward = await self._compute_rewards(resp, prompt_str, task_data)

    # Log single scalar - appends to internal list
    # `workflow_context.stat_scope()` automatically differentiates evaluation/training scopes
    stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

    return resp, reward
```

You can log any other scalars in your customized workflow, e.g.,

```python
async def run(self, data, **extra_kwargs):
    # `workflow_context.stat_scope()` automatically differentiates evaluation/training scopes
    stats_tracker.get(workflow_context.stat_scope()).scalar(num_turns=num_turns, max_tokens=max_tokens, reward=reward)
    return reward
```

**Controller aggregation:**

The `RolloutController` collects stats from all workers and computes weighted averages:

```python
# areal/infra/controller/rollout_controller.py
def export_stats(self) -> dict[str, float]:
    all_raw_stats = self._collective_rpc(method="export_stats")

    # Aggregate using counts as weights
    stats, counts = defaultdict(float), defaultdict(int)
    for raw_stats in all_raw_stats:
        for k, v in raw_stats.items():
            if k.endswith("__count"):
                counts[k] += v
            else:
                stats[k] += v * raw_stats.get(k + "__count", 0)

    # Compute weighted averages
    return {k: v / counts[k + "__count"] for k, v in stats.items()
            if counts.get(k + "__count", 0) > 0}
```

### Batch Metrics (Training Engines)

Training engines process data in synchronized batches across data-parallel ranks.
Metrics are logged as tensors with boolean masks, then reduced across all ranks at
export time.

**Characteristics:**

- Log entire batch tensors with denominator masks
- Support for per-token and per-sequence statistics
- All-reduce synchronization ensures consistent stats across ranks
- Multiple reduce types: `AVG_MIN_MAX`, `AVG`, `SUM`, `MIN`, `MAX`

**Example from `PPOActor`:**

```python
# areal/trainer/ppo/actor.py
def ppo_update(self, data):
    loss_mask = data["loss_mask"].bool()
    reward_score = data["rewards"]

    # Define denominators (boolean masks)
    stats_tracker.denominator(
        n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
        n_valid_tokens=loss_mask,
    )

    # Log tensor metrics with denominator reference
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

**Export behavior:**

```python
# areal/engine/fsdp_engine.py
def export_stats(self) -> dict[str, float]:
    # All-reduce across data-parallel group
    return stats_tracker.export_all(reduce_group=self.data_parallel_group)
    # All DP ranks receive identical results
```

## API Reference

### Recording Methods

| Method                        | Use Case                    | Example                                  |
| ----------------------------- | --------------------------- | ---------------------------------------- |
| `scalar(**kwargs)`            | Single float values         | `scalar(lr=0.001, eps=0.2)`              |
| `denominator(**kwargs)`       | Define boolean masks        | `denominator(valid=mask.bool())`         |
| `stat(denominator, **kwargs)` | Tensor metrics with masking | `stat(loss=tensor, denominator="valid")` |

### Reduce Types

When using `stat()`, metrics default to `AVG_MIN_MAX`, which creates three output keys:

```python
stats_tracker.stat(loss=tensor, denominator="valid")
# Exports: {"loss/avg": 0.5, "loss/min": 0.1, "loss/max": 0.9}
```

Available reduce types:

| Type          | Output                          | Description              |
| ------------- | ------------------------------- | ------------------------ |
| `AVG_MIN_MAX` | `key/avg`, `key/min`, `key/max` | Default for tensor stats |
| `AVG`         | `key`                           | Weighted average only    |
| `SUM`         | `key`                           | Sum across all elements  |
| `MIN`         | `key`                           | Minimum value            |
| `MAX`         | `key`                           | Maximum value            |
| `SCALAR`      | `key`, `key__count`             | For scalar values        |

### Scoping

Organize related metrics using hierarchical scopes:

```python
with stats_tracker.scope("ppo_actor"):
    with stats_tracker.scope("update"):
        stats_tracker.stat(loss=loss_tensor, denominator="valid")
        # Key: "ppo_actor/update/loss/avg"
```

### Timing

Measure execution time with automatic scoping under `timeperf/`:

```python
with stats_tracker.record_timing("rollout"):
    batch = actor.prepare_batch(dataloader, workflow)
# Key: "timeperf/rollout"
```

### Named Trackers

Isolate metrics for different components:

```python
# Training metrics (default tracker)
stats_tracker.scalar(grad_norm=1.5)

# Rollout metrics
stats_tracker.get("rollout").scalar(reward=0.8)

# Evaluation metrics
stats_tracker.get("eval-rollout").scalar(reward=0.9)

# Export from all trackers
all_stats = stats_tracker.export_all(reduce_group=group)
```

If `evaluator.eval_before_train: true`, evaluation runs once before the first training
step to get an estimate of the initial model's performance before finetuning.

## Data Flow

The complete metrics flow from collection to logging:

```
Rollout Workers                          Training Workers
───────────────                          ────────────────
workflow.arun_episode()                  actor.ppo_update(batch)
        │                                        │
        ▼                                        ▼
get("rollout").scalar(r=0.5)             stat(tensor, denom=mask)
        │                                        │
        ▼                                        ▼
export_stats(reduce_group=None)          export_stats(reduce_group=dp_group)
{reward: 0.5, reward__count: 1}          → all_reduce across DP ranks
        │                                        │
        ▼                                        │
RolloutController.export_stats()                 │
→ weighted avg across workers                    │
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

## StatsLogger: Logging Backends

The
[`StatsLogger`](https://github.com/areal-project/AReaL/blob/main/areal/utils/stats_logger.py)
sends aggregated metrics to external logging backends. It is automatically managed by
`PPOTrainer` and runs only on rank 0 to avoid duplicate logging.

### Supported Backends

| Backend              | Configuration                     | Description                     |
| -------------------- | --------------------------------- | ------------------------------- |
| **Weights & Biases** | `config.stats_logger.wandb`       | Cloud-based experiment tracking |
| **SwanLab**          | `config.stats_logger.swanlab`     | Alternative experiment tracking |
| **TensorBoard**      | `config.stats_logger.tensorboard` | Local visualization             |

### Integration with PPOTrainer

The trainer calls `StatsLogger.commit()` at the end of each training step:

```python
# areal/trainer/rl_trainer.py
def _export_and_commit_stats(self, epoch, epoch_step, global_step):
    # 1. Collect metrics from all components
    stats = self.actor.export_stats()           # Training metrics (all-reduced)
    stats.update(self.rollout.export_stats())   # Rollout metrics (controller-aggregated)
    stats.update(self.eval_rollout.export_stats())  # Eval metrics

    # 2. Send to logging backends (rank 0 only)
    self.stats_logger.commit(epoch, epoch_step, global_step, stats)
```

### StatsLogger.commit()

The `commit()` method filters out internal count keys and logs to all configured
backends:

```python
# areal/utils/stats_logger.py
def commit(self, epoch, step, global_step, data):
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # Only rank 0 logs

    # Filter out __count keys (used internally for weighted averaging)
    data = {k: v for k, v in data.items() if not k.endswith("__count")}

    # Log to all backends
    wandb.log(data, step=global_step)
    swanlab.log(data, step=global_step)
    if self.summary_writer:
        for key, val in data.items():
            self.summary_writer.add_scalar(key, val, global_step)
```

### Configuration

Configure logging backends in your experiment config:

```yaml
stats_logger:
  experiment_name: "gsm8k_grpo"
  trial_name: "run_001"
  fileroot: "/path/to/logs"

  wandb:
    mode: "online"  # "online", "offline", or "disabled"
    project: "my-project"
    entity: "my-team"

  swanlab:
    mode: "online"  # "online", "local", or "disabled"
    project: "my-project"

  tensorboard:
    path: "/path/to/tensorboard/logs"  # null to disable
```

## Best Practices

1. **Choose the right paradigm**: Use `scalar()` for scalars, `stat()` with denominators
   for batched pytorch tensors (usually training metrics).

1. **Define denominators first**: Always call `denominator()` before `stat()` to
   establish the masking relationship.

1. **Use named trackers**: Use
   `stats_tracker.get(workflow_context.stat_scope()).scalar(...)` to isolate rollout
   (`"rollout"`) and evaluation (`"eval-rollout"`) metrics from training metrics.

## Training throughput, estimated FLOPs, and MoE balance

Archon, Megatron, and FSDP export training work under `train_perf`. Each interval spans
all `train_batch` calls since the previous statistics export (including repeated PPO
updates). Tokens are the sum of the original attention masks: prompt and response both
count, masked padding does not. Counts are summed across data-parallel replicas only;
TP, CP/SP, PP and EP do not multiply the logical batch size.

Timing uses synchronized wall time around batch preparation, forward, backward and
optimizer work, starting before gradient zeroing. Rollout, reference/evaluation
forwards, loading the next batch, checkpointing and statistics export are excluded. The
denominator is the maximum accumulated training time across training ranks.

| Metric                                                | Meaning                                                 |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `train_perf/tokens`                                   | Global logical tokens processed in this export interval |
| `train_perf/seconds`                                  | Training time for this interval                         |
| `train_perf/tokens_per_second`                        | Interval tokens / interval seconds                      |
| `train_perf/estimated_flops`                          | Sum of per-sequence forward + backward FLOPs            |
| `train_perf/estimated_flops_per_second`               | Estimated interval FLOPs / interval seconds             |
| `train_perf/total_tokens`, `train_perf/total_seconds` | Totals since engine initialization                      |
| `train_perf/cumulative_tokens_per_second`             | Total tokens / total training time                      |
| `train_perf/cumulative_estimated_flops_per_second`    | Total estimated FLOPs / total training time             |

These are aggregate training-cluster rates, not per-GPU rates. Cumulative counters
restart when the engine is recreated, including checkpoint recovery. Unsupported model
architectures still log tokens and time, but omit FLOPs metrics.

### FLOPs registration and conventions

`areal.utils.flops` provides factories for `qwen3_moe` and `qwen3_5_moe[_text]`,
including Qwen3-30B-A3B and Qwen3.5-35B-A3B. The actual checkpoint's text configuration
supplies dimensions, so a local checkpoint directory works without name matching. The
reference dimensions are from the official
[Qwen3 configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
and
[Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json).

A multiply-add counts as two FLOPs. The estimate is three times forward work (forward
plus an estimated two-times-forward backward). It includes selected routed expert
projections, routers, shared experts and their gate, attention projections, and the
language-model head. Causal full attention includes
`2 * query_heads * head_dim * L * (L + 1)` forward FLOPs per full-attention layer. For
packed batches, sum `estimate(L_i)` over individual sequences; do not use
`estimate(sum(L_i))`.

Qwen3.5 accounts separately for its full-attention layers and linear-attention
GatedDeltaNet layers, including gated projections and depthwise convolution. DeltaNet
uses a recurrent-equivalent estimate of three key-by-value multiply-adds per value head
per token (state prediction, update and readout). Its work grows linearly with sequence
length. This is a model-math estimate, not a count of the extra operations used by a
particular chunked kernel implementation.

These estimates exclude activation recomputation, optimizer math, elementwise
operations, softmax/normalization, auxiliary MTP branches and the vision encoder. They
describe a full-parameter text-model training workload, not measured hardware FLOPs,
exact LoRA/frozen-parameter backward work, or vision-model FLOPs. Tree training reports
logical per-sequence work; shared-prefix savings are not subtracted.

Register a custom factory in **each training worker before engine initialization**:

```python
from areal.utils.flops import register_flops_estimator


def my_model_factory(config):
    # Return a callable whose only input is one sequence's length.
    active_parameters = config.active_parameters
    heads = config.num_attention_heads
    head_dim = config.head_dim
    layers = config.num_hidden_layers

    def total_training_flops(sequence_length: int) -> float:
        linear = 6 * active_parameters * sequence_length
        attention = 6 * layers * heads * head_dim * sequence_length * (sequence_length + 1)
        return float(linear + attention)

    return total_training_flops


register_flops_estimator("my_model_type", my_model_factory)
```

Registration replaces any existing factory for that model type. Registering only in a
controller process does not register it in remote training workers.

### MoE balance and W&B visualization

MoE diagnostics use the independent `moe_balance` scope. Archon supports its common MoE
module, Megatron supports `TopKRouter`, and FSDP supports the Transformers
`Qwen3MoeTopKRouter` and `Qwen3_5MoeTopKRouter` contracts. Decoder layer IDs are
zero-based and global across pipeline stages. Shared experts and auxiliary MTP routers
are excluded. Unsupported routers do not emit MoE diagnostics.

For each layer, let `c_e` be the accumulated routing assignments to expert `e`, and `E`
the number of routed experts:

- Expert load percentage: `100 * c_e / sum(c_e)`; one layer sums to 100%.
- Ideal load: `sum(c_e) / E`, including top-k assignment multiplicity.
- `moe_balance/layer_<id>/max_over_ideal`: `max(c_e) / (sum(c_e) / E)`; perfect balance
  is 1. Empty layers report zeros.

Counts are reduced before normalization, so unequal microbatches and data-parallel
shards are weighted correctly. Only original training forwards count; evaluation and
checkpoint backward recomputation do not. These are **executed routing loads**:
packing/alignment padding is included; Megatron counts its post-capacity/drop routing
map. Consequently routed counts can differ from logical throughput token counts.

W&B receives one `moe_balance/expert_loads` Table per logged step with columns `layer`,
`expert`, `tokens`, and `load_percent`. Per-layer `max_over_ideal` remains a scalar.
This avoids thousands of individual W&B expert scalar series. SwanLab and Trackio retain
`moe_balance/layer_<id>/expert_<id>/{tokens,load_percent}`. TensorBoard and the console
report layer summaries without individual expert scalar series.

Recommended panels, built with
[W&B custom charts](https://docs.wandb.ai/models/app/features/custom-charts):

1. **Layer × expert heatmap at a selected step:** color by `load_percent`, with tokens
   in the tooltip. Keep expert order fixed. For cross-model comparisons, derive
   `load_percent / (100 / E)` and center the color scale at 1.
1. **Step × layer heatmap:** color by `max_over_ideal` to locate when a layer becomes
   imbalanced. This uses the per-layer scalar history.
1. **Single-layer expert bar chart:** filter the table by layer and draw an ideal
   reference line at `100 / E`. Keep expert IDs on the x-axis to reveal persistent hot
   or idle experts.

Use `historyTable` and the step selector to switch snapshots directly. To display
multiple steps simultaneously in a Table-based chart, combine snapshots with their log
steps in postprocessing. Per-layer maximum load ratios already have scalar history.
Quantiles, entropy, and coefficient of variation can be derived from the raw loads.

See [MoE expert load visualization](moe_visualization.md) for the UI walkthrough, a
ready-to-paste Vega specification, and recovery from 10,000-row preview truncation.
