from pathlib import Path

import torch
import yaml

_CONFIG_PATH = Path("examples/hermes/config.yaml")


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_hermes_singleton_online_config_preserves_outcome_signal():
    config = _load_config()
    actor = config["actor"]

    assert config["gconfig"]["n_samples"] == 1
    assert config["train_dataset"]["batch_size"] == 1
    assert actor.get("reward_norm") is None
    assert actor.get("adv_norm") is None

    outcomes = torch.tensor([0.0, 1.0])
    signed_rewards = (outcomes + actor["reward_bias"]) * actor["reward_scaling"]
    flat_token_advantages = signed_rewards[:, None].expand(-1, 3)

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert torch.count_nonzero(flat_token_advantages).item() == 6
    assert not torch.equal(flat_token_advantages[0], flat_token_advantages[1])
