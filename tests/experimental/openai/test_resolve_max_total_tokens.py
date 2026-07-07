# SPDX-License-Identifier: Apache-2.0

from areal.experimental.openai.client import (
    _DEFAULT_MAX_TOTAL_TOKENS,
    _resolve_max_total_tokens,
)


def test_engine_max_tokens_caps_total_length():
    out = _resolve_max_total_tokens(
        prompt_len=1000,
        max_new_tokens=100_000,
        engine_max_tokens=4096,
    )

    assert out == 4096


def test_fallback_ceiling_applies_when_engine_max_tokens_is_none():
    out = _resolve_max_total_tokens(
        prompt_len=1000,
        max_new_tokens=100_000,
        engine_max_tokens=None,
    )

    assert out == _DEFAULT_MAX_TOTAL_TOKENS


def test_returns_prompt_plus_generation_when_under_cap():
    out = _resolve_max_total_tokens(
        prompt_len=100,
        max_new_tokens=200,
        engine_max_tokens=4096,
    )

    assert out == 300
