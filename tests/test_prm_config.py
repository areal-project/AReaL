"""Small CPU-only tests for the rollout.agent.prm configuration contract."""

from __future__ import annotations

import pytest

from areal.api.cli_args import (
    PPOConfig,
    PRMAdvantageShapingConfig,
    PRMScorerConfig,
)


def _config_with_prm() -> PPOConfig:
    config = PPOConfig()
    config.rollout.agent.prm.scorers = [
        PRMScorerConfig(path="tests.fake_scorers.ToolFormatScorer")
    ]
    return config


def test_prm_accepts_v1_concat_agent_config():
    config = _config_with_prm()
    config.rollout._version = "v1"
    config.rollout.agent.export_style = "concat"
    config.rollout.agent.chat_template_type = "concat"

    config.__post_init__()


def test_prm_rejects_folded_process_rewards_until_semantics_are_defined():
    """Configured scorers must not silently lose whole-turn reward magnitude."""
    config = _config_with_prm()
    config.rollout._version = "v1"
    config.rollout.agent.export_style = "concat"
    config.rollout.agent.chat_template_type = "concat"
    config.actor.token_rewards_as_adv = False

    with pytest.raises(ValueError, match="folded process rewards are not supported"):
        config.__post_init__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "v2", "rollout._version='v1'"),
        ("export_style", "individual", "export_style='concat'"),
        ("chat_template_type", "hf", "chat_template_type='concat'"),
    ],
)
def test_prm_rejects_unsupported_rollout_contract(field, value, message):
    config = _config_with_prm()
    config.rollout._version = "v1"
    config.rollout.agent.export_style = "concat"
    config.rollout.agent.chat_template_type = "concat"
    if field == "version":
        config.rollout._version = value
    else:
        setattr(config.rollout.agent, field, value)

    with pytest.raises(ValueError, match=message):
        config.__post_init__()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negative_scale", -0.1),
        ("zero_penalty", float("inf")),
        ("zero_eps", float("nan")),
    ],
)
def test_gvpo_config_rejects_invalid_coefficients(field, value):
    with pytest.raises(ValueError, match=field):
        PRMAdvantageShapingConfig(**{field: value})


def test_gvpo_requires_direct_process_advantage_mode():
    config = PPOConfig()
    config.rollout.agent.prm.advantage_shaping.mode = "gvpo"
    config.actor.token_rewards_as_adv = False

    with pytest.raises(ValueError, match="token_rewards_as_adv=True"):
        config.__post_init__()


def test_process_weighting_config_accepts_direct_process_advantage_mode():
    """The new shaping mode is valid with direct per-token process rewards."""
    shaping = PRMAdvantageShapingConfig(mode="process_weighted")
    config = PPOConfig()
    config.rollout.agent.prm.advantage_shaping = shaping

    config.__post_init__()


def test_process_weighting_config_rejects_folded_process_rewards():
    """Process weighting cannot fold process rewards into GAE first."""
    config = PPOConfig()
    config.rollout.agent.prm.advantage_shaping.mode = "process_weighted"
    config.actor.token_rewards_as_adv = False

    with pytest.raises(ValueError, match="token_rewards_as_adv=True"):
        config.__post_init__()


def test_process_weighting_config_rejects_mask_no_eos_with_zero():
    """Process weighting cannot reuse a zero introduced by no-EOS masking."""
    config = PPOConfig()
    config.rollout.agent.prm.advantage_shaping.mode = "process_weighted"
    config.actor.mask_no_eos_with_zero = True

    with pytest.raises(ValueError, match="mask_no_eos_with_zero=True"):
        config.__post_init__()


@pytest.mark.parametrize("mode", ["additive", "gvpo"])
def test_existing_shaping_modes_allow_mask_no_eos_with_zero(mode):
    """The incompatibility is limited to process-weighted shaping."""
    config = PPOConfig()
    config.rollout.agent.prm.advantage_shaping.mode = mode
    config.actor.mask_no_eos_with_zero = True

    config.__post_init__()


def test_prm_import_does_not_load_math_verify():
    """SWE rollout images can load PRM without the mathematics reward stack."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['math_verify'] = None; "
            "from areal.reward.prm import PRMRunner, PRMScorerResult",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
