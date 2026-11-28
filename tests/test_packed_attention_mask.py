# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoConfig, AutoModelForCausalLM

from areal.engine.fsdp_utils.attn_impl import (
    PACKED_MASK_ATTN_IMPLS,
    PACKED_SEQ_BOUNDS_KWARG,
    build_packed_attention_mask,
    register_packed_sdpa,
)

# Two packed sequences of length 2 and 3, plus one trailing padding position.
CU_SEQLENS = torch.tensor([0, 2, 5], dtype=torch.int32)
SEQ_LEN = 6
EXPECTED_ALLOWED = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=torch.bool,
)


def _build(attn_impl: str, **kwargs) -> torch.Tensor:
    return build_packed_attention_mask(
        CU_SEQLENS,
        SEQ_LEN,
        device=torch.device("cpu"),
        dtype=torch.float32,
        attn_impl=attn_impl,
        **kwargs,
    )


def test_sdpa_mask_is_block_diagonal_causal_boolean():
    mask = _build("sdpa")
    assert mask.shape == (1, 1, SEQ_LEN, SEQ_LEN)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0, 0], EXPECTED_ALLOWED)


def test_eager_mask_is_additive_with_dtype_min_on_blocked_entries():
    mask = _build("eager")[0, 0]
    assert mask.dtype == torch.float32
    assert torch.all(mask[EXPECTED_ALLOWED] == 0)
    assert torch.all(mask[~EXPECTED_ALLOWED] == torch.finfo(torch.float32).min)


def test_sliding_window_limits_lookback_within_a_sequence():
    mask = _build("sdpa", sliding_window=2)[0, 0]
    # Position 4 is the third token of the second sequence: with a window of 2 it
    # may see positions 3 and 4, but no longer position 2.
    assert mask[4].tolist() == [False, False, False, True, True, False]


def _tiny_qwen3(attn_impl: str) -> torch.nn.Module:
    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl)
    return model.eval()


def _packed_inputs(lengths: list[int], pad: int = 0):
    torch.manual_seed(1)
    seqs = [torch.randint(0, 128, (n,)) for n in lengths]
    cu_seqlens = torch.tensor([0, *lengths]).cumsum(0).to(torch.int32)
    input_ids = torch.cat([*seqs, torch.zeros(pad, dtype=torch.long)])[None]
    position_ids = torch.cat([*(torch.arange(n) for n in lengths), torch.arange(pad)])[
        None
    ]
    return seqs, cu_seqlens, input_ids, position_ids


@pytest.mark.parametrize("attn_impl", PACKED_MASK_ATTN_IMPLS)
def test_packed_forward_matches_separate_sequences(attn_impl):
    """Packed logits must equal running each sequence on its own.

    Uses the same mechanism FSDPEngine uses per backend: ``packed_seq_bounds``
    for sdpa, a block-diagonal mask for eager. Also asserts the unprotected packed
    forward disagrees, so the test fails if the protection stops being applied.
    """
    register_packed_sdpa()
    model = _tiny_qwen3(attn_impl)
    lengths = [5, 7, 4]
    seqs, cu_seqlens, input_ids, position_ids = _packed_inputs(lengths, pad=3)
    no_mask = dict(full_attention=None, sliding_attention=None)

    with torch.no_grad():
        separate = torch.cat([model(s[None]).logits[0] for s in seqs])
        if attn_impl == "sdpa":
            kwargs = {
                "attention_mask": no_mask,
                PACKED_SEQ_BOUNDS_KWARG: tuple(cu_seqlens.tolist()),
            }
        else:
            mask = build_packed_attention_mask(
                cu_seqlens,
                input_ids.shape[-1],
                device=torch.device("cpu"),
                dtype=torch.float32,
                attn_impl=attn_impl,
            )
            kwargs = {
                "attention_mask": dict(full_attention=mask, sliding_attention=mask)
            }
        packed = model(input_ids, position_ids=position_ids, **kwargs).logits[0]
        unprotected = model(
            input_ids, position_ids=position_ids, attention_mask=no_mask
        ).logits[0]

    n = int(cu_seqlens[-1])
    assert_close(packed[:n], separate, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(unprotected[:n], separate, rtol=1e-4, atol=1e-4)


def test_registered_sdpa_is_unchanged_without_packed_bounds():
    """Overriding the ``sdpa`` entry must not change calls that pass no bounds."""
    seqs, _, input_ids, _ = _packed_inputs([6, 5])
    model = _tiny_qwen3("sdpa")
    with torch.no_grad():
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import AttentionInterface

        AttentionInterface.register("sdpa", sdpa_attention_forward)
        before = model(input_ids).logits
        register_packed_sdpa()
        after = model(input_ids).logits
    assert torch.equal(before, after)
