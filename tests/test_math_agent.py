# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from examples.math.gsm8k_config import GSM8KGRPOConfig

from areal.workflow.openai import math_agent


def test_math_agent_configures_reward_worker_count(monkeypatch):
    """The reward pool size is configurable and is not sent to OpenAI."""
    reward_wrapper = Mock()
    monkeypatch.setattr(math_agent, "AsyncRewardWrapper", reward_wrapper)

    agent = math_agent.MathAgent(reward_max_workers=1, temperature=0.8)

    reward_wrapper.assert_called_once_with(
        math_agent.math_reward_fn,
        max_workers=1,
    )
    assert agent.kwargs == {"temperature": 0.8}


def test_gsm8k_config_rejects_nonpositive_reward_worker_count():
    """GSM8K rejects invalid reward worker limits during config loading."""
    with pytest.raises(ValueError, match="reward_max_workers must be positive"):
        GSM8KGRPOConfig(reward_max_workers=0)
