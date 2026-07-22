# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from areal.api import ModelResponse
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _response(
    *,
    input_tokens: list[int],
    output_tokens: list[int],
    routed: np.ndarray,
) -> ModelResponse:
    return ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[1] * len(output_tokens),
        routed_experts=routed,
    )


def test_interaction_to_tensor_dict_includes_r3_routing():
    routed = np.arange(1, 1 + 3 * 4, dtype=np.int32).reshape(3, 4)
    interaction = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            routed=routed,
        ),
        reward=1.0,
        r3_num_moe_layers=2,
        r3_topk=2,
    )

    tensor_dict = interaction.to_tensor_dict()

    assert tensor_dict["routed_experts"].shape == (1, 4, 2, 2)
    assert tensor_dict["r3_routing_valid"].tolist() == [True]
    torch.testing.assert_close(
        tensor_dict["routed_experts"][0, 3],
        tensor_dict["routed_experts"][0, 2],
        rtol=0,
        atol=0,
    )


def test_concat_interaction_to_tensor_dict_concatenates_parent_routing():
    parent_routed = np.arange(1, 1 + 2 * 4, dtype=np.int32).reshape(2, 4)
    child_routed = np.arange(101, 101 + 5 * 4, dtype=np.int32).reshape(5, 4)
    parent = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1],
            output_tokens=[2, 3],
            routed=parent_routed,
        ),
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    child = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2, 3, 4],
            output_tokens=[5, 6],
            routed=child_routed,
        ),
        parent=parent,
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )

    tensor_dict = child.to_tensor_dict()

    routed = tensor_dict["routed_experts"]
    assert routed.shape == (1, 6, 2, 2)
    torch.testing.assert_close(
        routed[0, :3],
        parent.to_tensor_dict()["routed_experts"][0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        routed[0, 3:],
        torch.as_tensor(
            np.stack(
                [
                    child_routed.reshape(5, 2, 2)[3],
                    child_routed.reshape(5, 2, 2)[4],
                    child_routed.reshape(5, 2, 2)[4],
                ]
            )
        ).int(),
        rtol=0,
        atol=0,
    )
    assert tensor_dict["r3_routing_valid"].tolist() == [True]


def test_proxy_interaction_serialization_preserves_r3_tensor_cache():
    routed = np.arange(1, 1 + 3 * 4, dtype=np.int32).reshape(3, 4)
    interaction = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            routed=routed,
        ),
        reward=1.0,
    )
    interaction.interaction_id = "completion-1"

    serialized = serialize_interactions(
        {"completion-1": interaction},
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    restored = deserialize_interactions(serialized)

    tensor_dict = restored["completion-1"].to_tensor_dict()
    assert tensor_dict["routed_experts"].shape == (1, 4, 2, 2)
    assert tensor_dict["r3_routing_valid"].tolist() == [True]
