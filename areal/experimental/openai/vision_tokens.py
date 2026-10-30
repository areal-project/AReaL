# SPDX-License-Identifier: Apache-2.0
"""Keep literal visual special tokens in chat text out of HF image expansion."""

from copy import copy, deepcopy
from typing import Any
from uuid import uuid4

from areal.utils.hf_utils import apply_chat_template

_VISUAL_TOKENS = (
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


def protect_vision_tokens(
    processor: Any,
    tokenizer: Any,
    messages: list[dict],
    tools: Any,
    template_kwargs: dict,
    prompt_text: str,
) -> tuple[Any, str, dict[int, int]]:
    """Escape request text using single tokens on a request-local tokenizer.

    Actual typed image parts are rendered by the template as usual. Restoring
    the temporary IDs after processing preserves token boundaries, including
    when literal visual tokens touch ordinary text. Shared tokenizer/processor
    objects and the canonical history are never mutated.
    """

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from strings(key)
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)

    # The template source contains genuine image placeholders and is code,
    # whereas its other arguments are request data rendered into that code.
    text_kwargs = {
        key: value for key, value in template_kwargs.items() if key != "chat_template"
    }
    values = (messages, tools, text_kwargs)
    text_values = list(strings(values))
    text_tokenizer = getattr(processor, "tokenizer", None)
    decoder = getattr(text_tokenizer, "added_tokens_decoder", {})
    dangerous = [
        token
        for token in _VISUAL_TOKENS
        if text_tokenizer is not None
        and any(token in s for s in text_values)
        and str(decoder.get(text_tokenizer.convert_tokens_to_ids(token))) == token
    ]
    if not dangerous:
        return processor, prompt_text, {}

    from transformers import AddedToken

    local_tokenizer = deepcopy(processor.tokenizer)
    replacements = {}
    restore_ids = {}
    for token in dangerous:
        original_id = local_tokenizer.convert_tokens_to_ids(token)
        original = local_tokenizer.added_tokens_decoder.get(original_id)
        if original is None or str(original) != token:
            raise ValueError(f"Cannot preserve unregistered visual token {token!r}")
        marker = f"<|areal_text_{uuid4().hex}|>"
        while marker in prompt_text or any(marker in s for s in text_values):
            marker = f"<|areal_text_{uuid4().hex}|>"
        local_tokenizer.add_tokens(
            [
                AddedToken(
                    marker,
                    **{
                        key: getattr(original, key)
                        for key in (
                            "single_word",
                            "lstrip",
                            "rstrip",
                            "normalized",
                            "special",
                        )
                    },
                )
            ],
            special_tokens=True,
        )
        marker_ids = local_tokenizer.encode(marker, add_special_tokens=False)
        if len(marker_ids) != 1:
            raise ValueError("Visual text escape must encode to exactly one token")
        replacements[token] = marker
        restore_ids[marker_ids[0]] = original_id

    def escape(value):
        if isinstance(value, str):
            for token, marker in replacements.items():
                value = value.replace(token, marker)
            return value
        if isinstance(value, dict):
            return {escape(key): escape(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [escape(item) for item in value]
        return value

    protected = apply_chat_template(
        tokenizer,
        escape(messages),
        tools=escape(tools),
        add_generation_prompt=True,
        tokenize=False,
        **{
            **escape(text_kwargs),
            **(
                {"chat_template": template_kwargs["chat_template"]}
                if "chat_template" in template_kwargs
                else {}
            ),
        },
    )
    restored = protected
    for token, marker in replacements.items():
        restored = restored.replace(marker, token)
    if restored != prompt_text:
        raise ValueError(
            "Chat template changes structure when escaping visual text tokens"
        )
    local_processor = copy(processor)
    local_processor.tokenizer = local_tokenizer
    return local_processor, protected, restore_ids
