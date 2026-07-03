from pathlib import Path

import torch
import yaml

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
    signed_rewards = (counterfactual_outcomes + actor["reward_bias"]) * actor[
        "reward_scaling"
    ]

    for signed_reward in signed_rewards:
        single_trajectory_token_advantages = signed_reward.expand(3)
        assert torch.count_nonzero(single_trajectory_token_advantages).item() == 3

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert not torch.equal(signed_rewards[0], signed_rewards[1])
