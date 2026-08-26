# SPDX-License-Identifier: Apache-2.0

"""The non-agent VLM workflow must build its collapsed prompt from the whole
rendered prompt, not from a slice of it."""

import pytest

from areal.utils.hf_utils import collapsed_prompt_token_ids


class _StubTokenizer:
    def __call__(self, text, padding=False, **kwargs):
        # One id per character makes a truncated prompt obvious by length.
        return {"input_ids": [[ord(c) for c in item] for item in text]}


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()


# Datasets render the prompt with apply_chat_template(..., tokenize=False), so
# ``data["messages"]`` is a single string. Indexing it yields one character.
RENDERED_PROMPT = "<|im_start|>user\n<image>describe<|im_end|>"


def test_collapsed_prompt_uses_the_whole_rendered_prompt():
    """Test that the rendered prompt string is tokenized in full."""
    result = collapsed_prompt_token_ids(_StubProcessor(), RENDERED_PROMPT)

    assert len(result) == len(RENDERED_PROMPT)


def test_indexing_the_rendered_prompt_would_truncate_it():
    """Test that the first-character mistake is detectable, not silent.

    ``data["messages"][0]`` selects one character rather than one message, which
    would ship a one-token prompt alongside the full expanded one.
    """
    truncated = collapsed_prompt_token_ids(_StubProcessor(), RENDERED_PROMPT[0])

    assert len(truncated) == 1
    assert len(truncated) != len(RENDERED_PROMPT)


@pytest.mark.parametrize("prompt", ["", "x", RENDERED_PROMPT])
def test_collapsed_prompt_round_trips_lengths(prompt):
    """Test that the helper preserves the prompt's token count."""
    assert len(collapsed_prompt_token_ids(_StubProcessor(), prompt)) == len(prompt)
