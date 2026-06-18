"""Tests for multi-EOS stop-token handling on ModelResponse.

Covers the stop-token set (explicit stop_token_ids ∪ tokenizer eos/pad),
end_with_stop, and output_tokens_without_stop — including the multi-EOS case
(several valid EOS ids) that a single tokenizer.eos_token_id cannot express.
"""

from dataclasses import dataclass

import pytest

from areal.api.io_struct import ModelResponse


@dataclass
class _FakeTokenizer:
    """Minimal tokenizer stub: ModelResponse only reads eos/pad token ids.

    A stub (not a real tokenizer) is deliberate — a real single-EOS tokenizer
    can't exercise the multi-EOS path under test.
    """

    eos_token_id: int | None = 1
    pad_token_id: int | None = 0


def _resp(output_tokens, *, stop_token_ids=None, stop_reason="stop", tokenizer=None):
    return ModelResponse(
        input_tokens=[9, 9],
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        tokenizer=tokenizer if tokenizer is not None else _FakeTokenizer(),
        stop_token_ids=stop_token_ids,
    )


def test_end_with_stop_unions_explicit_ids_and_tokenizer_eos_pad():
    """end_with_stop recognises explicit stop_token_ids AND tokenizer eos/pad.

    Asserts through the public property: each of the explicit ids (106, 50) and
    the tokenizer's eos (1) / pad (0) is treated as a trailing stop.
    """
    for last in (106, 50, 1, 0):
        assert _resp([5, last], stop_token_ids=[106, 50]).end_with_stop is True
    assert _resp([5, 7], stop_token_ids=[106, 50]).end_with_stop is False


def test_end_with_stop_true_for_multi_eos_id():
    """A multi-EOS id (in stop_token_ids, not tokenizer.eos) counts as a stop."""
    # 106 is a valid EOS for the model but tokenizer.eos_token_id is only 1.
    resp = _resp([7, 8, 106], stop_token_ids=[106, 50])
    assert resp.end_with_stop is True


def test_end_with_stop_false_when_last_token_not_a_stop():
    resp = _resp([7, 8, 9], stop_token_ids=[106])
    assert resp.end_with_stop is False


def test_end_with_stop_false_on_empty_output():
    resp = _resp([], stop_token_ids=[106])
    assert resp.end_with_stop is False


def test_without_stop_strips_trailing_multi_eos():
    """output_tokens_without_stop strips a trailing multi-EOS id."""
    resp = _resp([7, 8, 106], stop_token_ids=[106, 50])
    assert resp.output_tokens_without_stop == [7, 8]


def test_without_stop_falls_back_to_tokenizer_eos_when_no_explicit_ids():
    """With stop_token_ids=None, behaviour falls back to tokenizer eos/pad."""
    resp = _resp([7, 8, 1], stop_token_ids=None)  # 1 == tokenizer.eos
    assert resp.output_tokens_without_stop == [7, 8]


def test_without_stop_passthrough_for_length_and_abort():
    """No stripping / no error when generation stopped by length or abort."""
    for reason in ("length", "abort"):
        resp = _resp([7, 8, 9], stop_token_ids=[106], stop_reason=reason)
        assert resp.output_tokens_without_stop == [7, 8, 9]


def test_without_stop_raises_when_no_trailing_stop_but_stop_reason():
    """stop_reason=stop but output doesn't end in a stop id → diagnostic error."""
    resp = _resp([7, 8, 9], stop_token_ids=[106])
    with pytest.raises(ValueError, match="not in the stop set"):
        _ = resp.output_tokens_without_stop


def test_without_stop_raises_when_all_tokens_are_stops():
    resp = _resp([106, 106], stop_token_ids=[106])
    with pytest.raises(ValueError, match="All output_tokens are stop tokens"):
        _ = resp.output_tokens_without_stop


def test_without_stop_raises_on_empty_stop_set():
    """No tokenizer eos/pad and no explicit ids → cannot identify stops."""
    resp = _resp(
        [7, 8],
        stop_token_ids=None,
        tokenizer=_FakeTokenizer(eos_token_id=None, pad_token_id=None),
    )
    with pytest.raises(ValueError, match="Empty stop-token set"):
        _ = resp.output_tokens_without_stop
