"""Unit tests for SWE SFT thinking-mode classification."""

import importlib.util
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from areal.dataset.swe_sft.pipeline import _tokenize_samples


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
_patch_chat_template_for_training = tokenization._patch_chat_template_for_training
_render_tokenize_mask = tokenization._render_tokenize_mask


class _AdaptiveTokenizer:
    chat_template = (
        "enable_thinking preserved_thinking {% generation %}{% endgeneration %}"
    )

    def __init__(self):
        self.enable_thinking = None

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        self.enable_thinking = kwargs.get("enable_thinking")
        rendered = []
        mask = []
        for message in messages:
            content = message.get("content") or ""
            if message.get("role") == "assistant":
                header = "<role>ASSISTANT</role>"
                body = f"{content}<|role_end|>"
                rendered.append(header + body)
                mask.extend([0] * len(header) + [1] * len(body))
            else:
                rendered.append(content)
                mask.extend([0] * len(content))
        text = "".join(rendered)
        if not tokenize:
            return text
        assert kwargs["return_dict"] is True
        assert kwargs["return_assistant_tokens_mask"] is True
        return {"input_ids": list(range(len(text))), "assistant_masks": mask}

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


class _NativeMaskTokenizer:
    chat_template = "{% generation %}{% endgeneration %}"

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


class _JinjaCharTokenizer:
    """Render real Transformers chat templates with character-level tokens."""

    def __init__(self, chat_template):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        return_mask = kwargs.pop("return_assistant_tokens_mask", False)
        kwargs.pop("return_dict", None)
        tools = kwargs.pop("tools", None)
        rendered, generation_indices = render_jinja_template(
            conversations=[messages],
            tools=tools,
            chat_template=self.chat_template,
            return_assistant_tokens_mask=return_mask,
            **kwargs,
        )
        text = rendered[0]
        if not tokenize:
            return text

        mask = [0] * len(text)
        for start, end in generation_indices[0]:
            mask[start:end] = [1] * (end - start)
        return {"input_ids": list(range(len(text))), "assistant_masks": mask}

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


def _legacy_template(family):
    if family == "Qwen3":
        user_render = (
            "        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}\n"
        )
        old_block = tokenization._QWEN3_OLD_BLOCK
        assistant_end = "        {{- '<|im_end|>\\n' }}\n"
    else:
        user_render = (
            "        {{- '<role>HUMAN</role>' + message.content + '<|role_end|>' }}\n"
        )
        old_block = (
            tokenization._BAILING_CURRENT_OLD_BLOCK
            if family == "BailingCurrent"
            else tokenization._BAILING_OLD_BLOCK
        )
        assistant_end = "        {{- '<|role_end|>' }}\n"

    return (
        "{%- set ns = namespace(last_query_index=-1) %}\n"
        "{%- for message in messages %}\n"
        '    {%- if message.role == "user" %}\n'
        + user_render
        + '    {%- elif message.role == "assistant" %}\n'
        + "        {%- set content = message.content %}\n"
        + "        {%- set reasoning_content = '' %}\n"
        + "        "
        + old_block
        + "\n"
        + "        {%- if message.tool_calls %}\n"
        + "            {{- '<tool_call>' + message.tool_calls[0].function.name "
        + "+ '</tool_call>' }}\n"
        + "        {%- endif %}\n"
        + assistant_end
        + '    {%- elif message.role == "tool" %}\n'
        + "        {{- message.content }}\n"
        + "    {%- endif %}\n"
        + "{%- endfor %}"
    )


def _bailing_v3_template():
    return r"""{% set preserved_thinking = true %}
{%- set ns = namespace(last_query_index=-1) %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<role>HUMAN</role>' + message.content + '<|role_end|>' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- set content = message.content %}
        {%- if preserved_thinking or loop.index0 > ns.last_query_index %}
            {%- if reasoning_content != '' %}
                {{- '<role>ASSISTANT</role>' + '\n<think>' + reasoning_content.strip('\n') + '</think>' + content.lstrip('\n') }}
            {%- else %}
                {{- '<role>ASSISTANT</role>\n<think></think>' + content }}
            {%- endif %}
        {%- else %}
            {{- '<role>ASSISTANT</role>\n<think></think>' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {{- '<tool_call>' + message.tool_calls[0].function.name + '</tool_call>' }}
        {%- endif %}
        {{- '<|role_end|>' }}
    {%- elif message.role == "tool" %}
        {{- message.content }}
    {%- endif %}
{%- endfor %}"""


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


@pytest.mark.parametrize(
    ("family", "literal_eot", "assistant_header"),
    [
        ("Qwen3", "<|im_end|>", "<|im_start|>assistant\n"),
        ("Bailing", "<|role_end|>", "<role>ASSISTANT</role>"),
        ("BailingCurrent", "<|role_end|>", "<role>ASSISTANT</role>"),
    ],
)
@pytest.mark.parametrize("split_mode", ["pair", "trajectory"])
def test_training_patch_is_idempotent_and_masks_literal_delimiters(
    family,
    literal_eot,
    assistant_header,
    split_mode,
):
    tokenizer = _JinjaCharTokenizer(_legacy_template(family))
    payload = f"before {literal_eot} after"
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": payload,
            "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
        },
    ]
    rendered_before = tokenizer.apply_chat_template(messages, tokenize=False)

    _patch_chat_template_for_training(tokenizer)
    rendered_after = tokenizer.apply_chat_template(messages, tokenize=False)
    once_patched = tokenizer.chat_template
    _patch_chat_template_for_training(tokenizer)

    assert rendered_after == rendered_before
    assert tokenizer.chat_template == once_patched
    assert tokenizer.chat_template.count(tokenization._TRAINING_PATCH_MARKER) == 1

    result = _render_tokenize_mask(
        messages,
        tokenizer,
        split_mode=split_mode,
    )
    assert result is not None
    full_text, _, loss_mask, _ = result
    payload_start = full_text.index(payload)
    tool_call = "<tool_call>run</tool_call>"
    tool_call_start = full_text.index(tool_call)
    header_start = full_text.index(assistant_header)
    real_eot_start = full_text.rindex(literal_eot)
    assert not any(loss_mask[header_start : header_start + len(assistant_header)])
    assert all(loss_mask[payload_start : payload_start + len(payload)])
    assert all(loss_mask[tool_call_start : tool_call_start + len(tool_call)])
    assert all(loss_mask[real_eot_start : real_eot_start + len(literal_eot)])


def test_bailing_v3_training_patch_is_idempotent():
    tokenizer = _JinjaCharTokenizer(_bailing_v3_template())
    payload = "before <|role_end|> after"
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": payload,
            "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
        },
    ]
    rendered_before = tokenizer.apply_chat_template(messages, tokenize=False)

    _patch_chat_template_for_training(tokenizer)
    rendered_after = tokenizer.apply_chat_template(messages, tokenize=False)
    once_patched = tokenizer.chat_template
    _patch_chat_template_for_training(tokenizer)

    assert rendered_after == rendered_before
    assert tokenizer.chat_template == once_patched
    assert tokenizer.chat_template.count(tokenization._TRAINING_PATCH_MARKER) == 1

    result = _render_tokenize_mask(messages, tokenizer, split_mode="trajectory")
    assert result is not None
    full_text, _, loss_mask, _ = result
    header = "<role>ASSISTANT</role>"
    tool_call = "<tool_call>run</tool_call>"
    header_start = full_text.index(header)
    payload_start = full_text.index(payload)
    tool_call_start = full_text.index(tool_call)
    real_eot_start = full_text.rindex("<|role_end|>")
    assert not any(loss_mask[header_start : header_start + len(header)])
    assert all(loss_mask[payload_start : payload_start + len(payload)])
    assert all(loss_mask[tool_call_start : tool_call_start + len(tool_call)])
    assert all(loss_mask[real_eot_start : real_eot_start + len("<|role_end|>")])


def test_bailing_v3_training_patch_rejects_unverified_generation_layout():
    partial = _bailing_v3_template().replace(
        tokenization._BAILING_V3_ASSISTANT_START,
        '{%- elif message.role == "assistant" %}\n'
        "        {%- generation %}{%- endgeneration %}\n"
        "        {%- set reasoning_content = '' %}",
        1,
    )
    tokenizer = types.SimpleNamespace(chat_template=partial)

    with pytest.raises(ValueError, match="does not match the verified"):
        _patch_chat_template_for_training(tokenizer)


def test_render_tokenize_mask_rejects_delimiter_only_template():
    tokenizer = _JinjaCharTokenizer("{{ messages[0].content }}")

    with pytest.raises(ValueError, match="Jinja generation tracking"):
        _render_tokenize_mask(
            [{"role": "assistant", "content": "answer"}],
            tokenizer,
        )


def test_tokenize_samples_filters_rows_without_supervised_tokens():
    tokenizer = _NativeMaskTokenizer()
    messages = [
        [
            {"role": "user", "content": "bad task"},
            {"role": "assistant", "content": "bad answer"},
        ],
        [
            {"role": "user", "content": "good task"},
            {"role": "assistant", "content": "good answer"},
        ],
    ]

    dataset = _tokenize_samples(
        messages,
        [None, None],
        tokenizer,
        split_mode="trajectory",
        error_indices_list=[[0], []],
        num_proc=1,
    )

    assert len(dataset) == 1
    assert any(dataset[0]["loss_mask"])


def test_tokenize_samples_rejects_dataset_without_supervised_tokens():
    with pytest.raises(ValueError, match="no samples with a non-empty"):
        _tokenize_samples(
            [
                [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "answer"},
                ]
            ],
            [None],
            _NativeMaskTokenizer(),
            split_mode="trajectory",
            error_indices_list=[[0]],
            num_proc=1,
        )


def test_swe_sft_cli_help_starts_successfully():
    result = subprocess.run(
        [sys.executable, "-m", "areal.dataset.swe_sft", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Verify SWE SFT pair generation" in result.stdout
