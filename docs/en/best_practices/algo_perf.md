# Diagnosing RL Performance

This guide helps you diagnose and resolve common performance issues in reinforcement
learning training. Use the strategies below to identify bottlenecks, tune
hyperparameters, and optimize your RL workflows.

## Using Synchronous RL Instead of Asynchronous Training

If you suspect asynchronous RL training impacts learning performance, or if you want to
debug a new agentic application, you can switch to standard synchronous RL training with
the following configuration:

```yaml
rollout:
  max_head_offpolicyness: 0  # 0 implies synchronous training
actor:
  recompute_logprob: false  # use logprobs returned by inference backend
  use_decoupled_loss: false  # reverts to the original PPO loss
```

For detailed information about these configurations, see our
[asynchronous RL guide](../algorithms/async.md) and
[CLI reference](../cli_reference.md).

## Training Rewards Not Increasing

This is a common issue that may be due to multiple reasons. We recommend the following
diagnostic steps:

1. **Establish a baseline:** Run evaluation on the test set to measure baseline
   performance before training. AReaL allows zero-code changes between training and
   evaluation, so you can reuse your training code for evaluation. See
   [Evaluation Guide](../tutorial/eval.md) for details.
1. **Test on simpler data:** Run RL training on the test set instead of the training set
   to verify whether rewards increase.
1. **If rewards don't increase on the test set:** Tune your hyperparameters (e.g.,
   increase batch size or learning rate) or switch to a different base model. Consider
   applying SFT first, as this indicates the task may be too difficult for your current
   model.
1. **If rewards increase on test set but not training set:** Inspect the quality and
   difficulty of your training data. Ensure the distributions match and the difficulty
   is appropriate for your base model. You can enable dynamic filtering (similar to
   DAPO) by passing a `should_accept_fn` parameter to `prepare_batch` to ensure task
   difficulty remains appropriate during runtime. See our
   [detailed code walk-through](../tutorial/gsm8k_grpo.md) for more information.

## Important Metrics to Monitor

Monitoring these metrics helps ensure stable training and diagnose issues early.

### Reward Metrics

| Metric                  | Description                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval-rollout/reward`   | Reward on the **test set**. Primary indicator of model generalization.                                                                                                                              |
| `rollout/reward`        | Reward on the **training set**. Tracks learning progress during training.                                                                                                                           |
| `ppo/actor/task_reward` | Reward of trajectories **actually used for training**. Differs from `rollout/reward` when dynamic filtering is enabled—filtered trajectories are excluded here but still count in `rollout/reward`. |

**Troubleshooting high variance in `task_reward`:**

If `task_reward` fluctuates significantly, your training has high variance. Consider
increasing batch size if resources allow—this is often an effective remedy.

### Importance Weight Metrics

We recommend **asynchronous training with decoupled PPO loss**
(`use_decoupled_loss=true`) for optimal throughput. The two importance weight metrics
below are critical for monitoring training stability—their average values should remain
very close to 1.0.

With `use_decoupled_loss=true`, the loss function separates three policies:

- **π_behave**: Behavior policy that generated the samples during rollout
- **π_proximal**: Proximal policy, one training step behind the current policy
- **π_θ**: Current policy being optimized

The decoupled PPO loss combines two importance weights:

$$L = -\mathbb{E}\left[
\underbrace{\frac{\pi_{\text{proximal}}}{\pi_{\text{behave}}}}_{\text{behave imp weight}}
\cdot \min\left(
\underbrace{\frac{\pi_\theta}{\pi_{\text{proximal}}}}_{\text{importance weight}}
A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{proximal}}}, 1-\epsilon,
1+\epsilon\right) A \right) \right]$$

| Metric                               | Formula               | Description                                                       |
| ------------------------------------ | --------------------- | ----------------------------------------------------------------- |
| `ppo_actor/update/importance_weight` | π_θ / π_proximal      | Ratio for PPO clipping between current and proximal policies      |
| `ppo_actor/update/behave_imp_weight` | π_proximal / π_behave | Off-policy correction for distribution mismatch in async training |

**Troubleshooting `importance_weight` deviations:**

- If `importance_weight/avg` deviates significantly from 1, reduce `ppo_n_minibatches`.
- With `ppo_n_minibatches == 1`, theoretically `importance_weight` should equal 1
  exactly.
- If deviation persists at `ppo_n_minibatches == 1` (common in MoE training), add
  `actor.megatron.use_deterministic_algorithms=1` to your config.

**Troubleshooting `behave_imp_weight` deviations:**

- Ensure `rejection_sampling` is configured (e.g., `rejection_sampling: {level: token, action: mask, metric: ratio, upper: 5.0}`).
- If deviation persists, reduce `max_head_offpolicyness` to decrease sample staleness.

### Sequence Length Metrics

When training with long trajectories, monitor these metrics to detect truncation issues:

| Metric                   | Description                                                    |
| ------------------------ | -------------------------------------------------------------- |
| `ppo_actor/no_eos_ratio` | Fraction of trajectories truncated before generating EOS token |
| `ppo_actor/seq_len`      | Average sequence length during training                        |

**Troubleshooting high `no_eos_ratio`:**

If `no_eos_ratio` exceeds 0.05 (5% of trajectories truncated):

- Increase `max_new_tokens` to allow longer generations
- Use dynamic filtering to exclude overly long trajectories

**Monitoring sequence length growth:**

If `seq_len` increases steadily during training, watch `no_eos_ratio` closely—growing
sequence lengths often lead to more truncations.
