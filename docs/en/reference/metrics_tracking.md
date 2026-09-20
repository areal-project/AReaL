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

## Training Diagnostics

When recomputed proximal log-probabilities are available during advantage preprocessing,
`ppo_actor/train_infer/` compares them with the original rollout log-probabilities
**before** standard PPO replaces those values. This requires no additional model forward
pass. Runs without recomputation and pure MOPD preprocessing do not emit this namespace.

Let `d = log(p_trainer) - log(p_rollout)` on aligned, valid generated tokens:

| Metric                                                                                           | Definition                                  |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| `logp_diff/{avg,min,max}`                                                                        | Signed difference `d`                       |
| `logp_abs_diff/{avg,min,max}`                                                                    | Absolute difference `abs(d)`                |
| `logp_diff_squared`                                                                              | Token-weighted mean of `d²`                 |
| `trainer_nll`, `rollout_nll`                                                                     | Mean negative sampled-token log-probability |
| `kl_k1`, `kl_k2`, `kl_k3`                                                                        | Means of `-d`, `d²/2`, and `exp(d)-1-d`     |
| `ratio_outside_1.5`, `ratio_outside_2`, `ratio_outside_3`, `ratio_outside_5`, `ratio_outside_10` | Fraction with `abs(d) > log(threshold)`     |

These metrics use the generation mask before subsequent rejection/advantage masking. For
stale rollouts, differences include policy drift, not just numerical engine
disagreement. KL estimates also depend on the sampling distribution, filtering, and
log-probability convention. The `k1` sample estimate can be negative; `k2` is a local
approximation. The mean squared difference and absolute mean permit deriving the
population standard deviation of absolute differences after aggregation:
`sqrt(max(0, E[d²] - E[abs(d)]²))`.

The ratio tail definition matches the extreme token fraction in
[R3, Equation (3)](https://arxiv.org/html/2510.11370v1). For samples drawn from the
reported rollout distribution with matching support, `k1` and `k3` estimate
`KL(rollout || trainer)`. Returned logprobs are not necessarily those of the actual
sampling distribution after top-p/top-k filtering or greedy selection.

`nonfinite_logp_fraction` reports the fraction of valid tokens for which either logprob
is NaN or infinite. If any such token participates, ratio tail metrics are NaN rather
than silently classifying invalid comparisons as zero. Masked padding does not
contribute to this fraction or contaminate the tails. `kl_k3_overflow_fraction` reports
finite-input gaps whose `expm1` exceeds float64.

`ppo_actor/advantage_{positive,negative,zero}_fraction` uses the shaped input advantages
and batch loss mask **before M2/rejection filtering**. It measures token occupancy, not
task success or gradient signs. GSPO can further aggregate advantages by sequence inside
the loss.

`ppo_actor/update/version_stats/sample_staleness_{theta,proximal}_{avg,min,max}` uses
generated tokens with nonnegative versions. Averages are token-weighted and extrema are
global over the reduction group. Both legacy theta/proximal keys use the last published
checkpoint version: actor and rollout versions advance together after a PPO update. They
do not track intermediate optimizer minibatches. `stale_token_fraction` counts behavior
versions older than that checkpoint; `future_token_fraction` counts newer versions. Raw
rollout versions are shifted to prediction positions locally for this diagnostic,
including pure MOPD batches. This does not change the existing version-aware proximal
approximation used in the loss. The population is the batch **before M2/rejection
filtering**, counted once per `_ppo_update`, independently of minibatch forwards.
`n_valid_generated_tokens` sums these observations; reused trajectories in separate
updates are counted again. Empty populations omit averages and extrema rather than
reporting a fabricated zero.

New diagnostics use `stat_compact`: each metric stores only four float64 scalars (masked
sum, count, minimum, maximum) per batch until export. This avoids the former
approximately 106 bytes of retained diagnostic tensors per padded token. Existing PPO
statistics still retain their own full tensors, and the transient computation of
summaries still allocates per-token tensors. Long-context peak memory and runtime
overhead require workload-specific validation.

The optional loglinear proximal approximation still uses raw token-position rollout
versions and its original `current_version - 1` interpolation assumption. Those
pre-existing algorithm semantics require separate step-level validation; the
checkpoint-age diagnostics above are not evidence that they are corrected.

## Best Practices

1. **Choose the right paradigm**: Use `scalar()` for scalars, `stat()` with denominators
   for batched pytorch tensors (usually training metrics).

1. **Define denominators first**: Always call `denominator()` before `stat()` to
   establish the masking relationship.

1. **Use named trackers**: Use
   `stats_tracker.get(workflow_context.stat_scope()).scalar(...)` to isolate rollout
   (`"rollout"`) and evaluation (`"eval-rollout"`) metrics from training metrics.
