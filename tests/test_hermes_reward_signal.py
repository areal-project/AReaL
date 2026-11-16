from pathlib import Path

import torch
import yaml

from areal.api.cli_args import PPOActorConfig

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples/hermes/config.yaml"


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_hermes_singleton_online_config_preserves_each_outcome_signal():
    config = _load_config()
    actor = config["actor"]

    assert config["gconfig"]["n_samples"] == 1
    assert config["train_dataset"]["batch_size"] == 1
    assert actor.get("reward_norm") is None
    assert actor.get("adv_norm") is None

    counterfactual_outcomes = torch.tensor([0.0, 1.0])
    actor_defaults = PPOActorConfig()
    reward_bias = actor.get("reward_bias", actor_defaults.reward_bias)
    reward_scaling = actor.get("reward_scaling", actor_defaults.reward_scaling)
    signed_rewards = (counterfactual_outcomes + reward_bias) * reward_scaling

    for signed_reward in signed_rewards:
        single_trajectory_token_advantages = signed_reward.expand(3)
        assert torch.count_nonzero(single_trajectory_token_advantages).item() == 3

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert not torch.equal(signed_rewards[0], signed_rewards[1])
