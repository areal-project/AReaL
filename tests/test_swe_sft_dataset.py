"""Unit tests for SWE SFT thinking-mode classification."""

import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

import pytest


def _load_swe_modules():
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "areal" or name.startswith("areal.")
    }
    for name in list(sys.modules):
        if name == "areal" or name.startswith("areal."):
            del sys.modules[name]

    areal_module = types.ModuleType("areal")
    dataset_module = types.ModuleType("areal.dataset")
    dataset_module.__path__ = []
    swe_package = types.ModuleType("areal.dataset.swe_sft")
    swe_package.__path__ = []
    utils_module = types.ModuleType("areal.utils")
    utils_module.logging = logging
    areal_module.utils = utils_module
    sys.modules.setdefault("areal", areal_module)
    sys.modules.setdefault("areal.dataset", dataset_module)
    sys.modules.setdefault("areal.dataset.swe_sft", swe_package)
    sys.modules.setdefault("areal.utils", utils_module)

    package_path = Path(__file__).parents[1] / "areal" / "dataset" / "swe_sft"
    loaded = []
    try:
        for name in ("messages", "tokenization"):
            full_name = f"areal.dataset.swe_sft.{name}"
            spec = importlib.util.spec_from_file_location(
                full_name, package_path / f"{name}.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
    finally:
        for name in list(sys.modules):
            if name == "areal" or name.startswith("areal."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
    return loaded


messages, tokenization = _load_swe_modules()
_clean_message = messages._clean_message
_msg_has_thinking = messages._msg_has_thinking
_prepare_trajectory = messages._prepare_trajectory
_split_and_filter = messages._split_and_filter
_add_bailing_v3_generation_tags = tokenization._add_bailing_v3_generation_tags
_render_tokenize_mask = tokenization._render_tokenize_mask


class _AdaptiveTokenizer:
    chat_template = "enable_thinking preserved_thinking"

    def __init__(self):
        self.enable_thinking = None

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert tokenize is False
        self.enable_thinking = kwargs.get("enable_thinking")
        rendered = []
        for message in messages:
            content = message.get("content") or ""
            if message.get("role") == "assistant":
                rendered.append(f"<role>ASSISTANT</role>{content}<|role_end|>")
            else:
                rendered.append(content)
        return "".join(rendered)

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


class _NativeMaskTokenizer:
    chat_template = "{% generation %}"

    @staticmethod
    def _render(messages):
        text = ""
        mask = []
        for message in messages:
            content = message.get("content") or ""
            if message["role"] == "assistant":
                header = "<role>ASSISTANT</role>"
                body = content + "<|role_end|>"
                text += header + body
                mask.extend([0] * len(header) + [1] * len(body))
            else:
                body = content + "<|role_end|>"
                text += body
                mask.extend([0] * len(body))
        return text, mask

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text, mask = self._render(messages)
        if not tokenize:
            return text
        assert kwargs["return_dict"] is True
        assert kwargs["return_assistant_tokens_mask"] is True
        return {"input_ids": list(range(len(text))), "assistant_masks": mask}

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"role": "user", "content": "<think>reasoning</think>"}, False),
        ({"role": "assistant", "content": "answer"}, False),
        ({"role": "assistant", "content": "<think></think>answer"}, False),
        ({"role": "assistant", "content": "<thinking> \n </thinking>"}, False),
        (
            {
                "role": "assistant",
                "content": "<think></think><thinking>reasoning</thinking>",
            },
            True,
        ),
        ({"role": "assistant", "content": "", "reasoning_content": " \n "}, False),
        (
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reasoning",
            },
            True,
        ),
    ],
)
def test_msg_has_thinking_requires_non_empty_reasoning(message, expected):
    # Act
    result = _msg_has_thinking(message)

    # Assert
    assert result is expected


def test_split_and_filter_preserves_canonical_target_thinking():
    """Pair mode strips context thinking and preserves each target's thinking."""
    raw_messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<think>first</think>tool step"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "<think>second</think>summary"},
    ]

    pairs, n_errors, n_empty_calls, n_bare_calls = _split_and_filter(raw_messages)

    assert (n_errors, n_empty_calls, n_bare_calls) == (0, 0, 0)
    assert len(pairs) == 2
    assert "<think>first</think>" in pairs[0][-1]["content"]
    assert "<think>" not in pairs[1][1]["content"]
    assert "<think>second</think>" in pairs[1][-1]["content"]


def test_prepare_trajectory_preserves_thinking_and_masks_errors():
    """Trajectory mode retains canonical targets while masking error segments."""
    raw_messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<think>reason</think>call"},
        {"role": "tool", "content": "failed", "is_error": True},
        {"role": "user", "content": "recover"},
        {"role": "assistant", "content": "summary"},
    ]

    cleaned, masked, n_errors, n_empty_calls, n_bare_calls = _prepare_trajectory(
        raw_messages
    )

    assert masked == [0]
    assert (n_errors, n_empty_calls, n_bare_calls) == (1, 0, 0)
    assert "<think>reason</think>" in cleaned[1]["content"]
    assert cleaned[4]["content"].startswith("<think>\n</think>")


def test_render_tokenize_mask_sets_adaptive_mode_per_trajectory():
    # Arrange
    assistant_pattern = re.compile(
        r"<role>ASSISTANT</role>(.*?)<\|role_end\|>", re.DOTALL
    )
    empty_thinking = _clean_message(
        {"role": "assistant", "content": "answer"},
        strip_thinking=False,
        ensure_thinking=True,
    )
    real_thinking = _clean_message(
        {"role": "assistant", "content": "<think>reasoning</think>answer"},
        strip_thinking=False,
        ensure_thinking=True,
    )

    # Act
    no_thinking_tokenizer = _AdaptiveTokenizer()
    _render_tokenize_mask(
        [{"role": "user", "content": "task"}, empty_thinking],
        no_thinking_tokenizer,
        assistant_pattern,
        split_mode="trajectory",
    )
    mixed_tokenizer = _AdaptiveTokenizer()
    _render_tokenize_mask(
        [
            {"role": "user", "content": "task"},
            empty_thinking,
            real_thinking,
        ],
        mixed_tokenizer,
        assistant_pattern,
        split_mode="trajectory",
    )

    # Assert
    assert no_thinking_tokenizer.enable_thinking is False
    assert mixed_tokenizer.enable_thinking is True


def test_native_mask_ignores_literal_assistant_delimiter_and_masks_errors():
    # Arrange
    injected = "quoted <role>ASSISTANT</role> text"
    messages = [
        {"role": "user", "content": injected},
        {"role": "assistant", "content": "bad"},
        {"role": "assistant", "content": "good"},
    ]

    # Act
    result = _render_tokenize_mask(
        messages,
        _NativeMaskTokenizer(),
        re.compile(r"this fallback must not be used"),
        split_mode="trajectory",
        error_indices=[0],
    )

    # Assert
    full_text, _, loss_mask, _ = result
    injected_start = full_text.index(injected)
    bad_start = full_text.index("bad")
    good_start = full_text.index("good")
    assert not any(loss_mask[injected_start : injected_start + len(injected)])
    assert not any(loss_mask[bad_start : bad_start + len("bad")])
    assert all(loss_mask[good_start : good_start + len("good")])


def test_add_bailing_v3_generation_tags_keeps_header_outside_mask():
    # Arrange
    template = r"""{% set preserved_thinking = true %}
{%- if preserved_thinking or loop.index0 > ns.last_query_index %}{% endif %}
{%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {{- '<role>ASSISTANT</role>' + 'thinking' }}
        {{- '<role>ASSISTANT</role>\n<think></think>' + 'empty' }}
        {{- '<role>ASSISTANT</role>\n<think></think>' + 'content' }}
        {{- '<|role_end|>' }}
    {%- elif message.role == "tool" %}"""

    # Act
    patched = _add_bailing_v3_generation_tags(template)

    # Assert
    assert patched is not None
    assert "{{- '<role>ASSISTANT</role>' }}\n        {%- generation %}" in patched
    assert "{{- '<|role_end|>' }}\n        {%- endgeneration %}" in patched
    assert patched.count("<role>ASSISTANT</role>") == 1
