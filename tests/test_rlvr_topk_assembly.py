"""US-SC-VERIFY: Score Centering padding helpers (spec F-03).

Tests pad_topk_ids and pad_topk_logps - the helpers that ensure absent
top-k slots carry zero probability mass (id=0, logp=-inf => q=0).
"""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from areal.utils.score_centering import (
    NEG_INF_PAD,
    pad_topk_ids,
    pad_topk_logps,
)


def test_pad_topk_ids_pads_with_zero():
    """F-03: pad_topk_ids right-pads with id 0."""
    result = pad_topk_ids([5, 3], 4)
    assert result == [5, 3, 0, 0]


def test_pad_topk_ids_truncates_to_k():
    """F-03: pad_topk_ids truncates if input exceeds k."""
    result = pad_topk_ids([5, 3, 7, 1, 9], 3)
    assert result == [5, 3, 7]


def test_pad_topk_logps_pads_with_neg_inf():
    """F-03: pad_topk_logps right-pads with -inf (q=0, zero mass)."""
    result = pad_topk_logps([-0.1, -0.5], 4)
    assert result == [-0.1, -0.5, NEG_INF_PAD, NEG_INF_PAD]
    # Padded slots must contribute zero probability mass
    assert math.exp(result[2]) == 0.0
    assert math.exp(result[3]) == 0.0


def test_pad_topk_logps_truncates_to_k():
    """F-03: pad_topk_logps truncates if input exceeds k."""
    result = pad_topk_logps([-0.1, -0.5, -0.3, -0.7], 2)
    assert result == [-0.1, -0.5]


# === US-SC-VERIFY2: rlvr arun_episode tensor assembly (spec F-03) ===


def _make_workflow_with_mock_engine(resp):
    """Build an RLVRWorkflow with a mock tokenizer and dummy reward_fn.

    The workflow's __init__ calls gconfig.new_with_stop_and_pad_token_ids
    on the tokenizer; we patch that out so a bare MagicMock suffices.
    """
    from areal.api.cli_args import GenerationHyperparameters
    from areal.workflow.rlvr import RLVRWorkflow

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = "test output"

    gconfig = GenerationHyperparameters(max_new_tokens=4)
    # Patch new_with_stop_and_pad_token_ids to return self (no-op)
    gconfig.new_with_stop_and_pad_token_ids = lambda tok: gconfig

    wf = RLVRWorkflow.__new__(RLVRWorkflow)
    wf.tokenizer = mock_tokenizer
    wf.gconfig = gconfig
    wf.enable_thinking = False
    wf.get_input_ids_fn = lambda data, tok, eth: [1, 2]
    wf.data_extract_prompt_fn = lambda d: d.get("messages", "")
    wf.reward_fn = lambda *a, **kw: 0.0
    wf.async_reward_fn = AsyncMock(return_value=0.0)

    engine = MagicMock()
    engine.agenerate = AsyncMock(return_value=resp)
    return wf, engine


def test_arun_episode_assembles_sampler_topk_tensors():
    """F-03: arun_episode assembles sampler_topk tensors with shape [1, seq, k]."""
    from areal.api.io_struct import ModelResponse

    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        output_topk_ids=[[5, 3], [6, 4]],
        output_topk_logps=[[-0.1, -0.5], [-0.2, -0.4]],
    )
    wf, engine = _make_workflow_with_mock_engine(resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    assert "sampler_topk_token_ids" in result
    assert "sampler_topk_logps" in result
    assert result["sampler_topk_token_ids"].shape == (1, 4, 2)  # [1, 2+2, 2]
    assert result["sampler_topk_logps"].shape == (1, 4, 2)


def test_arun_episode_pads_input_positions_with_zero_and_neg_inf():
    """F-03: input positions padded with id=0 and logp=-inf (zero mass)."""
    import math

    from areal.api.io_struct import ModelResponse

    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        output_topk_ids=[[5, 3], [6, 4]],
        output_topk_logps=[[-0.1, -0.5], [-0.2, -0.4]],
    )
    wf, engine = _make_workflow_with_mock_engine(resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    ids = result["sampler_topk_token_ids"][0]  # [4, 2]
    logps = result["sampler_topk_logps"][0]  # [4, 2]
    # Input positions (0 and 1) should be all zeros / -inf
    assert ids[0].tolist() == [0, 0]
    assert ids[1].tolist() == [0, 0]
    assert logps[0].tolist() == [float("-inf"), float("-inf")]
    assert logps[1].tolist() == [float("-inf"), float("-inf")]
    # Padded slots contribute zero mass
    assert math.exp(logps[0][0].item()) == 0.0


def test_arun_episode_skips_topk_when_absent():
    """F-03: no output_topk -> result dict has no sampler_topk keys."""
    from areal.api.io_struct import ModelResponse

    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
    )
    wf, engine = _make_workflow_with_mock_engine(resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    assert "sampler_topk_token_ids" not in result
    assert "sampler_topk_logps" not in result


def test_arun_episode_handles_variable_length_topk():
    """F-03: variable-length top-k adapts k_dim = max(len)."""
    import math

    from areal.api.io_struct import ModelResponse

    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        output_topk_ids=[[5, 3], [6]],  # pos0: 2 items, pos1: 1 item
        output_topk_logps=[[-0.1, -0.5], [-0.2]],
    )
    wf, engine = _make_workflow_with_mock_engine(resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    ids = result["sampler_topk_token_ids"][0]  # [4, 2]
    logps = result["sampler_topk_logps"][0]  # [4, 2]
    assert ids.shape == (4, 2)  # k_dim = max(2, 1) = 2
    # Output position 0 (index 2): full top-k
    assert ids[2].tolist() == [5, 3]
    assert logps[2].tolist() == [pytest.approx(-0.1, abs=1e-5), -0.5]
    # Output position 1 (index 3): 1 real + 1 padded
    assert ids[3].tolist() == [6, 0]
    assert logps[3].tolist() == [pytest.approx(-0.2, abs=1e-5), float("-inf")]
    assert math.exp(logps[3][1].item()) == 0.0


# === END US-SC-VERIFY2 ===


# === US-SC-ENV1 T04.2: two-layer fix regression tests ===


def test_arun_episode_truncates_k_to_sc_top_k():
    """T04.2 Fix 1: when vLLM returns 1+N entries (sampled token + top-N),
    k_dim is capped at sc_top_k so the extra entry is truncated.

    Simulates vLLM v1 returning 129 entries (sc_top_k=128 + 1 sampled token)
    and verifies the assembled tensor has k=128, not 129.
    """
    import asyncio

    from areal.api.cli_args import GenerationHyperparameters
    from areal.api.io_struct import ModelResponse
    from areal.workflow.rlvr import RLVRWorkflow

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = "test output"

    sc_top_k = 4
    gconfig = GenerationHyperparameters(max_new_tokens=2, sc_top_k=sc_top_k)
    gconfig.new_with_stop_and_pad_token_ids = lambda tok: gconfig

    wf = RLVRWorkflow.__new__(RLVRWorkflow)
    wf.tokenizer = mock_tokenizer
    wf.gconfig = gconfig
    wf.enable_thinking = False
    wf.get_input_ids_fn = lambda data, tok, eth: [1, 2]
    wf.data_extract_prompt_fn = lambda d: d.get("messages", "")
    wf.reward_fn = lambda *a, **kw: 0.0
    wf.async_reward_fn = AsyncMock(return_value=0.0)

    # vLLM returns 1 + sc_top_k = 5 entries (sampled token + top-4)
    n_returned = sc_top_k + 1
    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        output_topk_ids=[
            list(range(n_returned)),
            list(range(n_returned)),
        ],
        output_topk_logps=[
            [-0.01 * i for i in range(n_returned)],
            [-0.01 * i for i in range(n_returned)],
        ],
    )
    engine = MagicMock()
    engine.agenerate = AsyncMock(return_value=resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    ids = result["sampler_topk_token_ids"]
    logps = result["sampler_topk_logps"]
    # k_dim must be sc_top_k (4), NOT n_returned (5)
    assert ids.shape[2] == sc_top_k, (
        f"Expected k_dim={sc_top_k}, got {ids.shape[2]} (vLLM 1+N not truncated)"
    )
    assert logps.shape[2] == sc_top_k


def test_arun_episode_falls_back_to_raw_max_k_when_sc_top_k_zero():
    """T04.2 Fix 1: when sc_top_k=0 (disabled), k_dim falls back to raw_max_k
    (max entries returned), preserving pre-fix behavior.
    """
    import asyncio

    from areal.api.cli_args import GenerationHyperparameters
    from areal.api.io_struct import ModelResponse
    from areal.workflow.rlvr import RLVRWorkflow

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = "test output"

    gconfig = GenerationHyperparameters(max_new_tokens=2, sc_top_k=0)
    gconfig.new_with_stop_and_pad_token_ids = lambda tok: gconfig

    wf = RLVRWorkflow.__new__(RLVRWorkflow)
    wf.tokenizer = mock_tokenizer
    wf.gconfig = gconfig
    wf.enable_thinking = False
    wf.get_input_ids_fn = lambda data, tok, eth: [1, 2]
    wf.data_extract_prompt_fn = lambda d: d.get("messages", "")
    wf.reward_fn = lambda *a, **kw: 0.0
    wf.async_reward_fn = AsyncMock(return_value=0.0)

    # 3 entries returned, no sc_top_k cap -> k_dim = 3
    resp = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        output_topk_ids=[[5, 3, 1], [6, 4, 2]],
        output_topk_logps=[[-0.1, -0.5, -0.3], [-0.2, -0.4, -0.6]],
    )
    engine = MagicMock()
    engine.agenerate = AsyncMock(return_value=resp)
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = asyncio.get_event_loop().run_until_complete(wf.arun_episode(engine, data))

    ids = result["sampler_topk_token_ids"]
    assert ids.shape[2] == 3, (
        f"Expected k_dim=3 (raw_max_k fallback), got {ids.shape[2]}"
    )


def test_concat_padded_tensors_uses_neg_inf_for_logprob_keys():
    """T04.2 Fix 2: concat_padded_tensors pads _logps/_logprobs keys with -inf
    (exp(-inf)=0, zero mass) instead of 0.0 (exp(0)=1, inflates q_mass).

    Simulates the cross-episode scenario: episode A has k=3, episode B has
    k=2. After concat, episode B's 3rd slot must be -inf (not 0.0), so it
    contributes zero probability mass.
    """
    from areal.utils.data import concat_padded_tensors

    # Episode A: k=3; Episode B: k=2 (needs 1 pad slot)
    dict_a = {
        "sampler_topk_token_ids": torch.tensor([[10, 20, 30]], dtype=torch.int32),
        "sampler_topk_logps": torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
    }
    dict_b = {
        "sampler_topk_token_ids": torch.tensor([[40, 50]], dtype=torch.int32),
        "sampler_topk_logps": torch.tensor([[-0.4, -0.5]], dtype=torch.float32),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
    }

    result = concat_padded_tensors([dict_a, dict_b])

    logps = result["sampler_topk_logps"]  # [2, 3]
    assert logps.shape == (2, 3)

    # Episode A (row 0): unchanged
    assert logps[0].tolist() == [
        pytest.approx(-0.1),
        pytest.approx(-0.2),
        pytest.approx(-0.3),
    ]
    # Episode B (row 1): 2 real + 1 padded with -inf (NOT 0.0)
    assert logps[1, 0].item() == pytest.approx(-0.4)
    assert logps[1, 1].item() == pytest.approx(-0.5)
    padded_val = logps[1, 2].item()
    assert padded_val == float("-inf"), (
        f"Padded logprob must be -inf (zero mass), got {padded_val}"
    )
    # The padded slot must contribute zero probability mass
    assert math.exp(padded_val) == 0.0

    # token_ids padded with 0 (id 0), not -inf
    ids = result["sampler_topk_token_ids"]
    assert ids[1, 2].item() == 0


# === END US-SC-ENV1 T04.2 ===
