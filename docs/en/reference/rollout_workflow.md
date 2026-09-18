# RolloutWorkflow Reference

This document describes the `RolloutWorkflow` abstraction, the core interface for
implementing rollout generation in AReaL's reinforcement learning pipeline.

**Notes**:

1. This page targets developers seeking a deep understanding of the codebase. For
   agentic RL training, use the high-level API described in the
   [Agentic RL Guide](../tutorial/agentic_rl.md).

1. **Legacy pattern**: Directly subclassing `RolloutWorkflow` is considered legacy and
   should not be used proactively. For new agentic RL workflows, use the
   [agent workflow pattern](./agent_workflow.md) with `async def run()` instead.

## Overview

A `RolloutWorkflow` defines how to generate training trajectories from input data. It
encapsulates the logic for:

- Tokenizing prompts and preparing model inputs
- Calling the inference engine to generate completions
- Computing rewards for generated outputs
- Packaging results into tensor dictionaries for training

## Interface

```python
from areal.api.workflow_api import RolloutWorkflow

class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow."""
        ...
```

### Parameters

| Parameter | Type              | Description                                     |
| --------- | ----------------- | ----------------------------------------------- |
| `engine`  | `InferenceEngine` | Inference engine for generating model responses |
| `data`    | `dict[str, Any]`  | A single sample from the dataloader             |

### Return Types

The `arun_episode` method supports three return types:

| Return Type                                 | Description                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `dict[str, torch.Tensor]`                   | Standard tensor format for training                                                                |
| `dict[str, InteractionWithTokenLogpReward]` | Token-level interactions (auto-converted to tensors); produced by the high-level `ArealOpenAI` API |
| `None`                                      | Rejected trajectory, excluded from training                                                        |

## Tensor Dictionary Format

When returning a tensor dictionary, the following fields are expected:

| Field            | Shape                   | Type    | Required | Description                         |
| ---------------- | ----------------------- | ------- | -------- | ----------------------------------- |
| `input_ids`      | `[batch_size, seq_len]` | int32   | Yes      | Token IDs (prompt + completion)     |
| `attention_mask` | `[batch_size, seq_len]` | bool    | Yes      | Valid token mask                    |
| `loss_mask`      | `[batch_size, seq_len]` | int32   | No       | Completion token mask (1 = train)   |
| `logprobs`       | `[batch_size, seq_len]` | float32 | No       | Log probabilities per token         |
| `rewards`        | `[batch_size]`          | float32 | No       | Per-sequence rewards                |
| `versions`       | `[batch_size, seq_len]` | int32   | No       | Weight version when token generated |

Example return value:

```python
return {
    "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
    "attention_mask": torch.ones(1, 5, dtype=torch.bool),
    "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
    "logprobs": torch.tensor([[0.0, 0.0, -0.5, -0.3, -0.2]], dtype=torch.float32),
    "rewards": torch.tensor([1.0], dtype=torch.float32),
    "versions": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
}
```

## Workflow Context

Inside `arun_episode`, access the execution context via the `workflow_context` module.
Each workflow instance has its own isolated context:

```python
from areal.infra import workflow_context

async def arun_episode(self, engine, data):
    # Get current execution context
    ctx = workflow_context.get()

    # Check if running in evaluation mode
    if ctx.is_eval:
        # Use different parameters for evaluation
        ...

    # Get task ID for logging
    task_id = ctx.task_id

    # Get stats scope based on mode ("rollout" or "eval-rollout")
    scope = workflow_context.stat_scope()
```

## Sample-level refill

Set `rollout.max_concurrent_samples` to a positive integer to bound concurrent complete
rollout episodes. For example, `max_concurrent_samples: 48` and group size 12 initially
admit four prompt groups. If each group finishes three episodes, the 12 released slots
admit a fifth group before any of the original groups finishes. A sample means one
complete rollout or agent episode, not one LLM request or tensor row. The whole next
group must fit; an oversized group raises `ValueError`.

The setting replaces the `max_concurrent_rollouts` group concurrency limit for
admission. Leaving it unset preserves group-level admission. `consumer_batch_size` and
accepted/running counters remain in prompt-group units; `max_head_offpolicyness` still
controls the version-based admission budget. A free sample slot does not release a
group's staleness budget. Pause and runner queue limits still apply. Direct distributed
executors divide the sample limit by the training data-parallel size; v1 and v2
controllers enforce a global limit.

The admission budget alone does not bound the age of a slow group that later groups keep
overtaking. When this setting is enabled, `PPOTrainer` also masks actor and critic loss
targets whose `current_version - token_version > max_head_offpolicyness`. Equality is
allowed; for example, at version 3 with a limit of 2, version-0 targets are masked while
version-1 targets remain eligible. Missing, unknown, or future versions on otherwise
trainable tokens are errors. Custom training loops must pass
`max_token_staleness=rollout.max_head_offpolicyness` to `ppo_update` themselves.

This is token-level objective filtering, not whole-group deletion or a promise that
every delivered trajectory is fresh. Complete groups still determine reward baselines
and advantages; masking happens afterwards, before optimizer minibatch packing.
Sequence-level importance ratios use the retained targets, and loss weights count those
same targets. Old tokens remain available as context; model auxiliary losses are not
filtered by this policy. All DP ranks skip a globally empty optimizer minibatch,
including its optimizer update; a locally empty rank still participates when other ranks
have valid targets. The outer training iteration, LR schedule and published version
continue to advance even if all optimizer minibatches are skipped.
`stale_token_fraction` reports the masked fraction and `stale_empty_minibatch` reports
skipped optimizer minibatches under the actor/critic metric scopes.

This follows the token-loss masking principle described in
[DeepSeek-V4.1 section 5.2.2](https://arxiv.org/html/2609.19969v1#S5.SS2.SSS2). The
integer version threshold here is an AReaL policy, not a published DeepSeek threshold.
Dataset concurrency controls and early-short-sample filtering from that report are
separate mechanisms. Throughput measurements with frozen weights cannot validate the
retained training-token throughput or quality under this filter.

Grouped workflows keep their gather, sample ordering, filtering and reward
normalization. Returning `None` releases the sample slot without creating training data.
v2 offline agents report each completed episode, including when group samples run
serially; not-yet-started serial members retain their reservations. Online v2 workflows
release their reservation when the delivered workflow completes.

Worker progress identifies the task, execution attempt and sample index. Duplicate and
stale-attempt notifications cannot release capacity twice. Remote submission is not
retried automatically in this mode: a lost response may hide a running group. A remote
timeout does not cancel the worker or free its unconfirmed slots. Late progress and
confirmed group completion can still free those slots; if the worker is lost
permanently, restart the rollout runtime rather than assuming its work has stopped.
Group failure drains sibling cancellation handlers before releasing their reservations.

Controller `export_stats()` includes live gauges `rollout/sample_inflight`,
`rollout/sample_capacity` and `rollout/partial_groups`. Partial groups have at least one
finished member and at least one unfinished member. Local executors expose the same
snapshot through `dispatcher.sample_stats()`.

## Trajectory Dumping

When `InferenceEngineConfig.dump_to_file=True`, trajectories are automatically saved to
disk for debugging and analysis.

### Configuration

```yaml
rollout:
  dump_to_file: true
  fileroot: "/path/to/logs"
  tokenizer_path: "model/tokenizer" # Required for text decoding
```

### Output Location

Trajectories are saved to:

```
{fileroot}/{experiment_name}/{trial_name}/[rollout|eval-rollout]/{version}/{task_id}.jsonl
```

Example:

```
/tmp/areal/my_exp/trial1/rollout/5/42.jsonl
```

### Output Format

Each line in the JSONL file contains:

```json
{
  "task_id": 42,
  "sample_idx": 0,
  "seqlen": 512,
  "prompt_len": 128,
  "head_version": 5,
  "tail_version": 6,
  "version_rle": [[5, 100], [6, 200]],
  "reward": 1.0,
  "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "completion": "The answer is 4.<|im_end|>"
}
```

**Field descriptions:**

| Field          | Description                                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_id`      | Batch task identifier                                                                                                                                                |
| `sample_idx`   | Index of the sample within the batch                                                                                                                                 |
| `seqlen`       | Effective sequence length                                                                                                                                            |
| `prompt_len`   | Index of the first generated token (`mask.index(1)`). For multi-turn agent rollouts this is the position of the first assistant generation, not `seqlen - sum(mask)` |
| `head_version` | Per-sample minimum model version among `loss_mask==1` tokens                                                                                                         |
| `tail_version` | Per-sample maximum model version among `loss_mask==1` tokens                                                                                                         |
| `version_rle`  | Run-length encoded per-token version sequence (output tokens only), e.g. `[[5, 100], [6, 200]]`                                                                      |
| `reward`       | Reward value for this sample                                                                                                                                         |
| `prompt`       | Decoded prompt text                                                                                                                                                  |
| `completion`   | Decoded completion text                                                                                                                                              |
| `segments`     | *(multi-turn only)* List of `{"role": "prompt"\|"gen"\|"context", "len": N, "text": "..."}`                                                                          |

**Directory naming:** The `{version}` directory is named by the batch-global maximum
version (`global_tail`). Individual records within the same directory may have
`tail_version <= global_tail`.

## Grouped Rollout

Grouped rollout runs the same workflow multiple times per input prompt, producing
diverse completions for training. This is useful for algorithms like GRPO that benefit
from multiple samples per prompt.

### Configuration

Set `group_size` when submitting rollouts:

```python
engine.submit(
    data=sample,
    workflow=MyWorkflow,
    workflow_kwargs={...},
    group_size=4,  # Run workflow 4 times per input
)
```

Or via CLI:

```yaml
rollout:
  group_size: 4
```

### How It Works

When `group_size > 1`, the workflow is wrapped in `GroupedRolloutWorkflow`:

1. The wrapper runs `arun_episode` concurrently `group_size` times using
   `asyncio.gather`
1. Results are merged based on their type:
   - **Tensor dictionaries**: Concatenated along the batch dimension
   - **InteractionWithTokenLogpReward dicts**: Merged into a single dictionary
1. Each slot returns its normal result type when usable and `None` when unusable. `None`
   is intentionally opaque: classification and retry policy stay in the producer.
1. The wrapper waits only for the original slots. It neither retries an unusable slot
   nor duplicates a usable result.
1. Usable slots are retained exactly once and concatenated. Their actual count remains a
   prompt-group boundary during reward and advantage normalization.
1. With `reward_normalization=True`, interaction rewards are normalized across the
   usable rollouts after the group passes the minimum-size filter.
   `drop_incomplete_group=True` still requires every original slot to succeed. A
   surviving rollout with a missing row reward causes the group to be dropped.
1. `min_usable_group_size` defaults to `1`. The v1 RL trainer sets it to `2` when reward
   or advantage normalization uses group statistics, because that statistic needs at
   least two observations; a singleton target group (`n_samples: 1`) is complete by
   definition and keeps the minimum of `1`. Setting `actor.min_usable_group_size`
   replaces this derived value; explicit values below `2` are rejected while group
   statistics are in use. Groups below the minimum return `None`; the asynchronous
   collector then takes another ready prompt group. Batch-relative PPO and REINFORCE
   retain a usable singleton.
1. Each v1 `arun_episode` call is one logical rollout. Context compaction and
   `agent.export_style: individual` can export multiple rows without aborting the group.
   The collector records contiguous row counts and optional reward references in
   `RolloutGroup`, under the `rollout_group` trajectory key. Batch concatenation moves
   this metadata into `TrajBatchMeta`; splitting restores it to trajectories. Usable
   group sizes count logical rollouts, including for `n_samples: 1`.
1. Reward normalization uses one reference per logical rollout, for both group and batch
   statistics. When a rollout's row rewards differ, the workflow must supply a finite
   `rollout_reward` scalar in its tensor dictionary, or on an exported
   `InteractionWithTokenLogpReward`. All supplied references within a rollout must
   agree. If omitted, equal row rewards provide the reference. The same centering and
   scaling apply to every row's own reward; leave-one-out excludes the entire logical
   rollout from its reference baseline. When the computed reference standard deviation
   is at or below normalization epsilon, centering is retained with a divisor of `1`.
   The built-in v1 agent workflow explicitly supplies its terminal reward for
   `individual` exports, preserving discounted row rewards and any reference already
   supplied. Custom workflows must declare their own reference for differing row
   rewards; no terminal, sum, or mean score is inferred.
1. An explicit reference is unchanged by the built-in row-length overlong penalty. Actor
   reward bias, scaling, and clipping apply to both rows and references. Without an
   explicit reference, penalized row rewards must still agree. Advantage normalization
   keeps its existing masked token statistics and per-token leave-one-out behavior;
   logical counts only select singleton fallbacks. GAE still runs separately on each
   row.

PPO-family actor loss remains globally token-weighted by default. Consequently, a
partial group with more valid response tokens contributes more loss weight than a
smaller or shorter group. This is the existing backward-compatible estimator, not an
implicit claim of equal prompt weighting.

Grouped rollouts export `target_slot_count`, `usable_slot_count`,
`trainable_slot_count`, `fully_masked_group`, `singleton_slot_group`,
`pre_filter_usable_slot_yield`, and `pre_filter_trainable_slot_yield`. These count the
original rollout calls; final accepted and rejected counts remain collector metrics
after `should_accept_fn` runs. PPO training separately reports the logical usable
group-size and valid-token loss-weight distributions, including per-size
`group_loss_weight_size_<N>` metrics.

### Output Shape

With `group_size=4`, a workflow returning `[1, seq_len]` tensors, and all four slots
usable, the grouped output has shape `[4, seq_len]`. An incomplete accepted group uses
its actual usable count as the leading dimension.

### Implementation

From `areal/infra/remote_inf_engine.py`:

```python
class GroupedRolloutWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # Run N times concurrently
        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data)
              for _ in range(self.group_size)]
        )

        # A normal result is usable; None is unusable.
        valid_results = [r for r in results if r is not None]
        if len(valid_results) < self.min_usable_group_size:
            return None

        # Merge based on result type
        if all_interaction_dicts(valid_results):
            return merge_dicts(valid_results)
        else:
            return concat_padded_tensors(valid_results)
```

## Implementing Custom Workflows

To create a custom workflow:

1. **Subclass `RolloutWorkflow`**:

```python
from areal.api.workflow_api import RolloutWorkflow

class MyWorkflow(RolloutWorkflow):
    def __init__(self, tokenizer, gconfig, **kwargs):
        self.tokenizer = tokenizer
        self.gconfig = gconfig

    async def arun_episode(self, engine, data):
        # 1. Prepare input
        input_ids = self.tokenizer.encode(data["prompt"])

        # 2. Generate completion
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig,
            tokenizer=self.tokenizer,
        )
        resp = await engine.agenerate(req)

        # 3. Compute reward
        reward = self.compute_reward(resp, data)

        # 4. Return tensor dict (or None to reject)
        if reward < 0:
            return None

        return self.build_tensor_dict(resp, reward)
```

2. **Register with trainer**:

```python
trainer.train(
    workflow=MyWorkflow,
    workflow_kwargs={
        "tokenizer": tokenizer,
        "gconfig": config.gconfig,
    },
)
```

## Workflow Resolution

Workflows can be specified in multiple ways:

| Format         | Example                          | Description                |
| -------------- | -------------------------------- | -------------------------- |
| Instance       | `MyWorkflow(...)`                | Pre-instantiated workflow  |
| Class          | `MyWorkflow`                     | Class (requires kwargs)    |
| String path    | `"my_module.MyWorkflow"`         | Dynamic import             |
| Agent workflow | Any class with `async def run()` | Wrapped with proxy support |

The training system automatically resolves these to `RolloutWorkflow` instances.

## See Also

- [Agentic RL Tutorial](../tutorial/agentic_rl.md) - Training with agent frameworks
- [Adding Custom Workflows](../customization/agent.md) - Step-by-step guide
