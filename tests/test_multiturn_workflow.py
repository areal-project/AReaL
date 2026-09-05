# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters, PPOCriticConfig
from areal.infra.workflow_executor import check_trajectory_format
from areal.trainer.ppo.critic import PPOCritic
from areal.workflow.multi_turn import (
    DEFAULT_MULTI_TURN_RETRY_PROMPT,
    MultiTurnWorkflow,
)


def _make_dummy_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"
    tokenizer.eos_token_id = 0
    return tokenizer


@pytest.mark.asyncio
async def test_multiturn_workflow_single_turn_rewards():
    tokenizer = _make_dummy_tokenizer()
    workflow = object.__new__(MultiTurnWorkflow)
    workflow.tokenizer = tokenizer
    workflow.gconfig = GenerationHyperparameters(max_new_tokens=4)
    workflow.max_turns = 3
    workflow.turn_discount = 0.9
    workflow.multi_turn_prompt_ids = [99]
    workflow.async_reward_fn = AsyncMock(return_value=1.0)

    engine = MagicMock()
    engine.agenerate = AsyncMock(
        return_value=ModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            output_logprobs=[0.0, 0.0],
            output_versions=[0, 0],
            stop_reason="stop",
        )
    )

    with (
        patch("areal.workflow.multi_turn.apply_chat_template", return_value=[1, 2]),
        patch("areal.workflow.multi_turn.stats_tracker"),
        patch("areal.workflow.multi_turn.workflow_context"),
    ):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        traj = await workflow.arun_episode(engine, data)

    # 1 turn executed
    assert engine.agenerate.await_count == 1
    assert traj["rewards"].shape == (1,)
    assert traj["rewards"].item() == 1.0
    assert traj["original_rewards"].shape == (1,)
    assert traj["original_rewards"].item() == 1.0

    # turn_rewards has 1 element
    assert traj["turn_rewards"].shape == (1, 1)
    torch.testing.assert_close(traj["turn_rewards"], torch.tensor([[1.0]]))

    # step_rewards has seq length (4 tokens: 2 prompt + 2 output)
    assert traj["step_rewards"].shape == (1, 4)
    expected_step_rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(traj["step_rewards"], expected_step_rewards)


@pytest.mark.asyncio
async def test_multiturn_workflow_multi_turn_reward_attribution():
    tokenizer = _make_dummy_tokenizer()
    workflow = object.__new__(MultiTurnWorkflow)
    workflow.tokenizer = tokenizer
    workflow.gconfig = GenerationHyperparameters(max_new_tokens=4)
    workflow.max_turns = 3
    workflow.turn_discount = 0.8
    workflow.multi_turn_prompt_ids = [99]
    # Turn 0 fails (0.0), Turn 1 succeeds (1.0)
    workflow.async_reward_fn = AsyncMock(side_effect=[0.0, 1.0])

    engine = MagicMock()
    engine.agenerate = AsyncMock(
        side_effect=[
            ModelResponse(
                input_tokens=[1, 2],
                output_tokens=[3],
                output_logprobs=[0.0],
                output_versions=[0],
                stop_reason="stop",
            ),
            ModelResponse(
                input_tokens=[1, 2, 3, 0, 99],
                output_tokens=[4, 5],
                output_logprobs=[0.0, 0.0],
                output_versions=[0, 0],
                stop_reason="stop",
            ),
        ]
    )

    with (
        patch("areal.workflow.multi_turn.apply_chat_template", return_value=[1, 2]),
        patch("areal.workflow.multi_turn.stats_tracker"),
        patch("areal.workflow.multi_turn.workflow_context"),
    ):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        traj = await workflow.arun_episode(engine, data)

    # 2 turns executed
    assert engine.agenerate.await_count == 2
    # Discounted reward: 1.0 * 0.8 = 0.8
    assert pytest.approx(traj["rewards"].item()) == 0.8
    # Original raw reward: 1.0
    assert pytest.approx(traj["original_rewards"].item()) == 1.0

    # turn_rewards has 2 entries: [0.0, 0.8]
    assert traj["turn_rewards"].shape == (1, 2)
    torch.testing.assert_close(traj["turn_rewards"], torch.tensor([[0.0, 0.8]]))

    # step_rewards has reward attributed to the final action token
    assert traj["step_rewards"].shape[0] == 1
    assert pytest.approx(traj["step_rewards"][0, -1].item()) == 0.8
    assert pytest.approx(traj["step_rewards"][0, :-1].sum().item()) == 0.0


def test_multiturn_workflow_custom_retry_prompt():
    tokenizer = _make_dummy_tokenizer()
    with patch("areal.workflow.multi_turn.apply_chat_template", return_value=[10, 20, 30]):
        # Default prompt
        wf_default = MultiTurnWorkflow(
            reward_fn=lambda *args, **kwargs: 1.0,
            gconfig=GenerationHyperparameters(),
            tokenizer=tokenizer,
            max_turns=2,
            turn_discount=1.0,
        )
        assert wf_default.retry_prompt == DEFAULT_MULTI_TURN_RETRY_PROMPT

        # Custom prompt
        custom_prompt = "Try fixing the syntax error in your function."
        wf_custom = MultiTurnWorkflow(
            reward_fn=lambda *args, **kwargs: 1.0,
            gconfig=GenerationHyperparameters(),
            tokenizer=tokenizer,
            max_turns=2,
            turn_discount=1.0,
            retry_prompt=custom_prompt,
        )
        assert wf_custom.retry_prompt == custom_prompt


def test_check_trajectory_format_exempts_turn_rewards():
    logger = MagicMock()
    # Batch with max_seqlen=5, but turn_rewards is [1, 2] (2 turns)
    data = {
        "input_ids": torch.zeros((1, 5), dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "turn_rewards": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }
    check_trajectory_format(data, logger=logger)
    # Ensure no warning was logged for turn_rewards having shape != max_seqlen
    for call in logger.warning.call_args_list:
        assert "turn_rewards" not in str(call)


def test_critic_pops_all_reward_metadata():
    engine = MagicMock()
    engine.train_batch.return_value = {}
    critic = PPOCritic(PPOCriticConfig(backend="fsdp:d1", ppo_n_minibatches=1), engine)
    data = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        "values": torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        "returns": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.int32),
        "rewards": torch.tensor([1.0]),
        "original_rewards": torch.tensor([1.0]),
        "turn_rewards": torch.tensor([[0.0, 1.0]]),
        "step_rewards": torch.tensor([[0.0, 0.0, 1.0]]),
    }
    critic._ppo_update(data)
    # Ensure all reward metadata keys were removed before staging
    assert "turn_rewards" not in data
    assert "step_rewards" not in data
    assert "original_rewards" not in data
    assert "rewards" not in data
