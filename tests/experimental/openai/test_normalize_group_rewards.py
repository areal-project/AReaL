# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from areal.api import ModelResponse
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    normalize_group_rewards,
)


def _make_dummy_completion(cid: str, content: str = "ok") -> ChatCompletion:
    return ChatCompletion(
        id=cid,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content=content, role="assistant"),
            )
        ],
        created=1000,
        model="test-model",
        object="chat.completion",
    )


def _make_dummy_response(input_tokens, output_tokens) -> ModelResponse:
    return ModelResponse(
        input_tokens=list(input_tokens),
        output_tokens=list(output_tokens),
        output_logprobs=[0.0] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
        stop_reason="stop",
    )


class TestNormalizeGroupRewards:
    def test_multi_turn_preserves_intermediate_step_rewards(self):
        # Rollout 1: Turn 1 (-0.2 step reward), Turn 2 (+1.0 outcome reward)
        i1_1 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("r1_t1"),
            reward=-0.2,
            model_response=_make_dummy_response([1, 2], [3]),
        )
        i1_2 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("r1_t2"),
            reward=1.0,
            parent=i1_1,
            model_response=_make_dummy_response([1, 2, 3, 4], [5]),
        )
        r1 = {"r1_t1": i1_1, "r1_t2": i1_2}

        # Rollout 2: Turn 1 (+0.1 step reward), Turn 2 (0.0 outcome reward)
        i2_1 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("r2_t1"),
            reward=0.1,
            model_response=_make_dummy_response([1, 2], [3]),
        )
        i2_2 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("r2_t2"),
            reward=0.0,
            parent=i2_1,
            model_response=_make_dummy_response([1, 2, 3, 4], [5]),
        )
        r2 = {"r2_t1": i2_1, "r2_t2": i2_2}

        # Call to_tensor_dict before normalization to populate _cache
        i1_1.to_tensor_dict()
        i1_2.to_tensor_dict()
        i2_1.to_tensor_dict()
        i2_2.to_tensor_dict()

        success = normalize_group_rewards([r1, r2])
        assert success is True

        # Rollout 1:
        # Intermediate turn r1_t1 must preserve its step reward (-0.2)
        assert pytest.approx(i1_1.reward) == -0.2
        assert pytest.approx(i1_1.original_reward) == -0.2
        assert pytest.approx(float(i1_1._cache["rewards"].item())) == -0.2
        assert pytest.approx(float(i1_1._cache["original_rewards"].item())) == -0.2

        # Terminal turn r1_t2 was normalized: mean=0.5, std=0.5 -> +1.0
        assert pytest.approx(i1_2.reward) == 1.0
        assert pytest.approx(i1_2.original_reward) == 1.0
        assert pytest.approx(float(i1_2._cache["rewards"].item())) == 1.0
        assert pytest.approx(float(i1_2._cache["original_rewards"].item())) == 1.0

        # Rollout 2:
        # Intermediate turn r2_t1 must preserve its step reward (+0.1)
        assert pytest.approx(i2_1.reward) == 0.1
        assert pytest.approx(i2_1.original_reward) == 0.1
        assert pytest.approx(float(i2_1._cache["rewards"].item())) == 0.1

        # Terminal turn r2_t2 was normalized: mean=0.5, std=0.5 -> -1.0
        assert pytest.approx(i2_2.reward) == -1.0
        assert pytest.approx(i2_2.original_reward) == 0.0
        assert pytest.approx(float(i2_2._cache["rewards"].item())) == -1.0
        assert pytest.approx(float(i2_2._cache["original_rewards"].item())) == 0.0

    def test_existing_original_reward_not_overwritten(self):
        # Suppose turn already has an original_reward set (e.g. from prior discounting)
        i = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("c1"),
            reward=2.5,
            original_reward=1.0,
            model_response=_make_dummy_response([1], [2]),
        )
        i.to_tensor_dict()
        r1 = {"c1": i}

        i2 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("c2"),
            reward=0.5,
            original_reward=0.5,
            model_response=_make_dummy_response([1], [2]),
        )
        i2.to_tensor_dict()
        r2 = {"c2": i2}

        success = normalize_group_rewards([r1, r2])
        assert success is True

        # Existing original_reward (1.0) must not be overwritten by 2.5
        assert i.original_reward == 1.0
        assert float(i._cache["original_rewards"].item()) == 1.0

    def test_single_turn_exact_parity(self):
        i1 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("c1"),
            reward=3.0,
            model_response=_make_dummy_response([1], [2]),
        )
        i2 = InteractionWithTokenLogpReward(
            completion=_make_dummy_completion("c2"),
            reward=1.0,
            model_response=_make_dummy_response([1], [2]),
        )
        i1.to_tensor_dict()
        i2.to_tensor_dict()

        success = normalize_group_rewards([{"c1": i1}, {"c2": i2}])
        assert success is True

        # mean = 2.0, std = 1.0 -> c1 = +1.0, c2 = -1.0
        assert pytest.approx(i1.reward) == 1.0
        assert pytest.approx(i2.reward) == -1.0
        assert pytest.approx(i1.original_reward) == 3.0
        assert pytest.approx(i2.original_reward) == 1.0

    def test_empty_or_incomplete_group_returns_false(self):
        assert normalize_group_rewards([]) is False
        assert normalize_group_rewards([None]) is False
        assert normalize_group_rewards([{}]) is False
        i_no_rew = InteractionWithTokenLogpReward(completion=_make_dummy_completion("c1"))
        assert normalize_group_rewards([{"c1": i_no_rew}]) is False
