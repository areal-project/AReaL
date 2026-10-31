# SPDX-License-Identifier: Apache-2.0

"""End-to-end sampler top-k positioning: parser, assembly, shift and packing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters, PPOActorConfig
from areal.engine.vllm_remote import VLLMBackend
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.trainer.ppo import actor as actor_module
from areal.utils.data import TrajBatchMeta, concat_padded_tensors, pack_tensor_dict
from areal.workflow.rlvr import RLVRWorkflow

POSITION_TOKEN_SETS = [[2, 0], [1, 3], [0, 2]]
POSITION_LOGPS = [[-0.4, -1.2], [-0.7, -2.1], [-0.2, -0.9]]


def _pickleable_reward_fn(prompt, completion, input_tokens, output_tokens, **task):
    """Module-level reward stub: AsyncRewardWrapper requires picklable callables."""
    return 1.0


def _parsed_response(prompt_len=2):
    """The real parser must attach one distinct head per generated position."""
    output_tokens = [21, 22, 23]
    raw = {
        "choices": [
            {
                "finish_reason": "stop",
                "token_ids": output_tokens,
                "logprobs": {
                    "content": [{"logprob": 0.0}] * 3,
                    "top_logprobs": [
                        {
                            f"token:{token}": float(logp)
                            for token, logp in zip(tokens, logps)
                        }
                        for tokens, logps in zip(POSITION_TOKEN_SETS, POSITION_LOGPS)
                    ],
                },
            }
        ],
    }
    response = VLLMBackend().parse_generation_response(raw)
    model_response = ModelResponse(
        input_tokens=list(range(14, 14 + prompt_len)),
        output_tokens=output_tokens,
        output_logprobs=[0.0] * 3,
        output_versions=[7] * 3,
        output_topk_ids=response.output_topk_ids,
        output_topk_logps=response.output_topk_logps,
    )
    return response, model_response


def test_parser_keeps_each_position_distinct_head():
    """Every generated position must keep its own token/logprob pairing."""
    response, _ = _parsed_response()
    assert response.output_topk_ids == POSITION_TOKEN_SETS
    assert response.output_topk_logps == POSITION_LOGPS
    for row in response.output_topk_logps:
        assert row[0] != row[1]


def test_openai_assembly_positions_prompts_and_outputs_hand_written():
    """Prompt rows must be empty and output rows must match the response layout."""
    _, model_response = _parsed_response()
    assembled = InteractionWithTokenLogpReward(
        sc_top_k=2,
        reward=0.0,
        model_response=model_response,
    ).to_tensor_dict()

    assert assembled["sampler_topk_token_ids"].shape == (1, 5, 2)
    torch.testing.assert_close(
        assembled["sampler_topk_token_ids"].squeeze(0),
        torch.tensor([[0, 0], [0, 0]] + POSITION_TOKEN_SETS, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    expected_logps = [[float("-inf")] * 2] * 2 + POSITION_LOGPS
    torch.testing.assert_close(
        assembled["sampler_topk_logps"].squeeze(0),
        torch.tensor(expected_logps, dtype=torch.float32),
        rtol=0,
        atol=1e-6,
    )


def test_assembly_caps_k_dim_and_rejects_short_heads():
    """A token plus N heads caps k; fewer entries than sc_top_k must fail."""
    _, model_response = _parsed_response()
    model_response.output_topk_ids[0].insert(0, 5)
    model_response.output_topk_logps[0].insert(0, -3.0)
    capped = InteractionWithTokenLogpReward(
        sc_top_k=2,
        reward=0.0,
        model_response=model_response,
    ).to_tensor_dict()
    assert capped["sampler_topk_token_ids"].shape == (1, 5, 2)
    # The inserted sampled-token entry caps k_dim to sc_top_k; its payload
    # must be truncated, keeping the response head intact per position.
    # The cap keeps the first sc_top_k raw entries (sampled token first):
    # response ordering, not a re-sorted top-k, survives truncation.
    assert capped["sampler_topk_token_ids"][0, 2].tolist() == [5, 2]
    assert capped["sampler_topk_logps"][0, 2, 0].item() == -3.0
    assert capped["sampler_topk_token_ids"][0, 3].tolist() == POSITION_TOKEN_SETS[1]

    short = ModelResponse(
        input_tokens=[10],
        output_tokens=[21],
        output_logprobs=[0.0],
        output_versions=[7],
        output_topk_ids=[[4]],
        output_topk_logps=[[-0.2]],
    )
    with pytest.raises(RuntimeError, match="sc_top_k=2"):
        InteractionWithTokenLogpReward(
            sc_top_k=2,
            reward=0.0,
            model_response=short,
        ).to_tensor_dict()


@pytest.mark.asyncio
async def test_rlvr_assembly_matches_openai_layout():
    """The RLVR rollout must place the same heads at the same token rows."""
    _, model_response = _parsed_response()

    workflow = RLVRWorkflow(
        reward_fn=_pickleable_reward_fn,
        gconfig=GenerationHyperparameters(max_new_tokens=8, sc_top_k=2),
        tokenizer=SimpleNamespace(
            eos_token_id=9,
            pad_token_id=0,
            decode=lambda ids: "stub",
        ),
        get_input_ids_fn=lambda data, tokenizer, enable_thinking: [14, 15],
    )
    engine = SimpleNamespace(agenerate=AsyncMock(return_value=model_response))
    result = await workflow.arun_episode(
        engine, {"messages": [{"role": "user", "content": "x"}]}
    )

    assert result["sampler_topk_token_ids"].shape == (1, 5, 2)
    torch.testing.assert_close(
        result["sampler_topk_token_ids"].squeeze(0),
        torch.tensor([[0, 0], [0, 0]] + POSITION_TOKEN_SETS, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    expected_logps = [[float("-inf")] * 2] * 2 + POSITION_LOGPS
    torch.testing.assert_close(
        result["sampler_topk_logps"].squeeze(0),
        torch.tensor(expected_logps, dtype=torch.float32),
        rtol=0,
        atol=1e-6,
    )


def test_actor_shift_aligns_logits_with_next_token_sampler_rows():
    """Token t must read sampler row t+1 exactly once, watching the shift."""
    prompt_rows = [[float("-inf"), float("-inf")]] * 2
    rows = prompt_rows + POSITION_LOGPS
    data = {
        "input_ids": torch.arange(5).reshape(1, 5),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.zeros(1, 5),
        "versions": torch.zeros(1, 5, dtype=torch.int32),
        "rewards": torch.zeros(1),
        "sampler_topk_logps": torch.tensor([rows], dtype=torch.float32),
        "sampler_topk_token_ids": torch.tensor(
            [[0, 0], [0, 0]] + POSITION_TOKEN_SETS,
            dtype=torch.int32,
        ).unsqueeze(0),
    }
    engine = SimpleNamespace(get_version=lambda: 0)
    actor = actor_module.PPOActor(PPOActorConfig(path="dummy"), engine)
    updated = actor._compute_advantages(
        data,
        TrajBatchMeta(n_trajs=1, traj_group_sizes=[1], traj_seqlens=[5]),
    )

    expected_logps = prompt_rows[1:] + POSITION_LOGPS + [prompt_rows[0]]
    torch.testing.assert_close(
        updated["sampler_topk_logps"],
        torch.tensor([expected_logps]),
        rtol=0,
        atol=1e-6,
    )
    assert updated["sampler_topk_token_ids"][0, 1].tolist() == POSITION_TOKEN_SETS[0]
    assert updated["sampler_topk_token_ids"][0, 4].tolist() == [0, 0]


def test_packing_preserves_per_position_pairing():
    """Packed rows must keep the exact sampler entries of their source tokens."""
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1]]),
        "logprobs": torch.zeros(1, 3),
        "sampler_topk_logps": torch.tensor([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]])
        .log()
        .unsqueeze(0),
        "sampler_topk_token_ids": torch.tensor([[2, 0], [1, 3], [0, 2]]).unsqueeze(0),
    }
    second = {
        "input_ids": torch.tensor([[6, 7, 8, 9]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[1, 1, 0, 0]]),
        "logprobs": torch.zeros(1, 4),
        "sampler_topk_logps": torch.tensor(
            [[0.14, 0.28], [0.42, 0.56], [0.7, 0.84], [0.98, 1.12]]
        )
        .log()
        .unsqueeze(0),
        "sampler_topk_token_ids": torch.tensor(
            [[0, 2], [2, 0], [1, 3], [0, 2]]
        ).unsqueeze(0),
    }
    packed = pack_tensor_dict(concat_padded_tensors([first, second]))

    assert packed["cu_seqlens"].tolist() == [0, 3, 7]
    assert packed["sampler_topk_logps"].shape == (7, 2)
    packed_logps = packed["sampler_topk_logps"][:, 0].exp().tolist()
    assert packed_logps == [
        pytest.approx(v, abs=1e-6) for v in (0.2, 0.6, 1.0, 0.14, 0.42, 0.7, 0.98)
    ]
    assert packed["sampler_topk_token_ids"][1].tolist() == [1, 3]
    assert packed["sampler_topk_token_ids"][4].tolist() == [2, 0]
