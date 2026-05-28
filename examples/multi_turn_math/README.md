# Training a Multi-Turn GSM8K Math Agent in AReaL

Files in this folder presents an example that train a multi-turn GSM8K math agent from
Qwen/Qwen2.5-1.5B-Instruct, using `ArealOpenAI` APIs and its `concat` mode to organize
training data and discount reward.

# To run the example

```bash
python3 examples/multi_turn_math/gsm8k_rl_mt.py \
    --config examples/multi_turn_math/gsm8k_grpo_mt.yaml \
    scheduler.type=ray \
    experiment_name=gsm8k-grpo-multiturn trial_name=trial0
```

only the following config are added compared to the original `gsm8k_grpo.yaml` config:

```yaml
export_style: concat
agent_run_args:
  max_turns: 2
```

## Reward Curve

<img align="center" alt="reward curve" src="reward_curve.png" width="100%">
