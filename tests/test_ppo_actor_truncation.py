from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import (
    GenerationHyperparameters,
    PPOActorConfig,
    PPOCriticConfig,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.critic import PPOCritic
from areal.utils.data import concat_padded_tensors
from areal.workflow.multi_turn import MultiTurnWorkflow
from areal.workflow.rlvr import RLVRWorkflow


def _make_trajectory(
    *,
    prompt_len: int,
    completion_len: int,
    reward: float,
    bootstrap_value: float,
    is_truncated: bool,
) -> dict[str, torch.Tensor]:
    seq_len = prompt_len + completion_len
    values = torch.zeros((1, seq_len), dtype=torch.float32)
    values[0, -1] = bootstrap_value

    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
        "loss_mask": torch.tensor(
            [[0] * prompt_len + [1] * completion_len], dtype=torch.bool
        ),
        "logprobs": torch.zeros((1, seq_len), dtype=torch.float32),
        "turn_ids": torch.tensor(
            [[-1] * prompt_len + [0] * completion_len], dtype=torch.int32
        ),
        "values": values,
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "is_truncated": torch.tensor([is_truncated], dtype=torch.bool),
    }


def _assert_action_advantages(output: dict[str, torch.Tensor], expected: float) -> None:
    action_advantages = output["advantages"][output["loss_mask"].bool()]
    torch.testing.assert_close(
        action_advantages,
        torch.full_like(action_advantages, expected),
        rtol=0,
        atol=0,
    )


def test_interaction_tensor_dict_preserves_length_termination():
    """Export the inference stop reason as per-trajectory training metadata."""
    for stop_reason, expected in (("length", True), ("stop", False)):
        response = ModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3],
            output_logprobs=[0.0],
            output_versions=[0],
            stop_reason=stop_reason,
        )
        interaction = InteractionWithTokenLogpReward(model_response=response)

        is_truncated = interaction.to_tensor_dict()["is_truncated"]

        torch.testing.assert_close(
            is_truncated,
            torch.tensor([expected], dtype=torch.bool),
            rtol=0,
            atol=0,
        )


@pytest.mark.asyncio
async def test_rlvr_workflow_preserves_length_termination():
    """Export the response stop reason from the built-in RLVR workflow."""
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"

    workflow = object.__new__(RLVRWorkflow)
    workflow.reward_fn = MagicMock()
    workflow.tokenizer = tokenizer
    workflow.enable_thinking = False
    workflow.gconfig = GenerationHyperparameters(max_new_tokens=3)
    workflow.get_input_ids_fn = lambda *_args: [1, 2]
    workflow.data_extract_prompt_fn = lambda data: data["messages"]

    for stop_reason, expected in (("length", True), ("stop", False)):
        response = ModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3],
            output_logprobs=[0.0],
            output_versions=[0],
            stop_reason=stop_reason,
        )
        workflow._collect_samples = AsyncMock(return_value=(response, 1.0))

        trajectory = await workflow.arun_episode(
            MagicMock(), {"messages": [{"role": "user", "content": "test"}]}
        )

        torch.testing.assert_close(
            trajectory["is_truncated"],
            torch.tensor([expected], dtype=torch.bool),
            rtol=0,
            atol=0,
        )


@pytest.mark.asyncio
async def test_multi_turn_uses_final_response_termination():
    """Let a later successful turn supersede an earlier length stop."""
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"
    tokenizer.eos_token_id = 0

    workflow = object.__new__(MultiTurnWorkflow)
    workflow.tokenizer = tokenizer
    workflow.gconfig = GenerationHyperparameters(max_new_tokens=3)
    workflow.max_turns = 2
    workflow.turn_discount = 1.0
    workflow.multi_turn_prompt_ids = [9]
    workflow.async_reward_fn = AsyncMock(side_effect=[0.0, 1.0])

    engine = MagicMock()
    engine.agenerate = AsyncMock(
        side_effect=[
            ModelResponse(
                input_tokens=[1, 2],
                output_tokens=[3],
                output_logprobs=[0.0],
                output_versions=[0],
                stop_reason="length",
            ),
            ModelResponse(
                input_tokens=[1, 2, 3, 0, 9],
                output_tokens=[4],
                output_logprobs=[0.0],
                output_versions=[0],
                stop_reason="stop",
            ),
        ]
    )

    with (
        patch("areal.workflow.multi_turn.apply_chat_template", return_value=[1, 2]),
        patch("areal.workflow.multi_turn.stats_tracker"),
        patch("areal.workflow.multi_turn.workflow_context"),
    ):
        trajectory = await workflow.arun_episode(
            engine, {"messages": [{"role": "user", "content": "test"}]}
        )

    assert engine.agenerate.await_count == 2
    torch.testing.assert_close(
        trajectory["is_truncated"],
        torch.tensor([False], dtype=torch.bool),
        rtol=0,
        atol=0,
    )


def test_critic_update_does_not_forward_truncation_metadata():
    """Remove trajectory metadata before constructing critic microbatches."""
    engine = MagicMock()
    engine.train_batch.return_value = {}
    critic = PPOCritic(PPOCriticConfig(backend="fsdp:d1", ppo_n_minibatches=1), engine)
    data = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.bool),
        "values": torch.zeros((2, 3), dtype=torch.float32),
        "returns": torch.zeros((2, 3), dtype=torch.float32),
        "is_truncated": torch.tensor([False, True], dtype=torch.bool),
    }

    critic._ppo_update(data)

    assert "is_truncated" not in data
    engine.train_batch.assert_called()
    for call in engine.train_batch.call_args_list:
        assert "is_truncated" not in call.args[0]


def test_ppo_update_logs_explicit_truncation_metadata():
    """Report the same per-trajectory truncation state used for advantages."""
    engine = MagicMock()
    engine.train_batch.return_value = {}
    actor = PPOActor(
        PPOActorConfig(
            backend="fsdp:d1",
            ppo_n_minibatches=1,
            mask_no_eos_with_zero=True,
            kl_ctl=0.0,
        ),
        engine,
    )
    outputs = actor.compute_advantages(
        [
            _make_trajectory(
                prompt_len=2,
                completion_len=3,
                reward=1.0,
                bootstrap_value=0.0,
                is_truncated=True,
            ),
            _make_trajectory(
                prompt_len=4,
                completion_len=3,
                reward=1.0,
                bootstrap_value=0.0,
                is_truncated=False,
            ),
        ]
    )
    batch = concat_padded_tensors(outputs)

    with patch("areal.trainer.ppo.actor.stats_tracker") as tracker:
        actor._ppo_update(batch)

    truncation_call = next(
        call for call in tracker.stat.call_args_list if "no_eos_ratios" in call.kwargs
    )
    torch.testing.assert_close(
        truncation_call.kwargs["no_eos_ratios"],
        torch.tensor([1.0, 0.0]),
        rtol=0,
        atol=0,
    )
    engine.train_batch.assert_called()
    for call in engine.train_batch.call_args_list:
        assert "is_truncated" not in call.args[0]


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_compute_advantages_uses_termination_reason_across_variable_padding(
    gae_timestep_unit: str,
):
    """Use explicit termination state, not the batch's padded sequence width."""
    actor = PPOActor(
        PPOActorConfig(
            backend="fsdp:d1",
            max_new_tokens=3,
            mask_no_eos_with_zero=True,
            kl_ctl=0.0,
            discount=1.0,
            gae_lambda=1.0,
            gae_timestep_unit=gae_timestep_unit,
        ),
        MagicMock(),
    )

    # The shorter trajectory was truncated, while the longest trajectory stopped
    # normally. Padding width therefore gives exactly the opposite classification.
    outputs = actor.compute_advantages(
        [
            _make_trajectory(
                prompt_len=2,
                completion_len=3,
                reward=3.0,
                bootstrap_value=10.0,
                is_truncated=True,
            ),
            _make_trajectory(
                prompt_len=4,
                completion_len=3,
                reward=3.0,
                bootstrap_value=20.0,
                is_truncated=False,
            ),
        ]
    )

    truncated, stopped = outputs

    # A truncated trajectory receives no task reward and bootstraps from its own
    # last valid token, even when it is shorter than the padded batch width.
    torch.testing.assert_close(
        truncated["tot_rewards"],
        torch.zeros_like(truncated["tot_rewards"]),
        rtol=0,
        atol=0,
    )
    _assert_action_advantages(truncated, expected=10.0)

    # A normally stopped trajectory keeps its task reward and must not bootstrap,
    # even when it happens to be the longest sequence in the batch.
    assert torch.count_nonzero(stopped["tot_rewards"]) == 1
    torch.testing.assert_close(
        stopped["tot_rewards"].sum(),
        torch.tensor(3.0),
        rtol=0,
        atol=0,
    )
    _assert_action_advantages(stopped, expected=3.0)
