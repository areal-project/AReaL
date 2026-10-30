# SPDX-License-Identifier: Apache-2.0

"""Image token transport shared by the SGLang inference adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest

IMAGE_SPANS_KEY = "areal_image_spans"


def image_prompt_metadata(
    metadata: dict[str, Any] | None,
    input_ids: list[int],
    token_types: list[int] | None,
    image_data: list[str] | None,
    processor: Any,
) -> dict[str, Any]:
    """Record inclusive real-image spans separately from same-ID text tokens."""
    result = dict(metadata or {})
    image_id = getattr(processor, "image_token_id", None)
    if not image_data or not isinstance(image_id, int) or token_types is None:
        return result
    spans = []
    for index, (token, modality) in enumerate(zip(input_ids, token_types, strict=True)):
        if token != image_id or modality != 1:
            continue
        if spans and spans[-1][1] == index - 1:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    if len(spans) != len(image_data):
        raise ValueError("Real image spans do not match the supplied images")
    result[IMAGE_SPANS_KEY] = spans
    return result


def image_request_kwargs(
    req: ModelRequest, accumulated_tokens: Sequence[int] = ()
) -> dict[str, Any]:
    """Use explicit offsets if text contains literal visual special tokens.

    Normal requests retain native compact-placeholder processing. The rare
    ambiguous case sends original IDs and real spans to AReaL's external Qwen
    processor, avoiding decode/retokenize and preserving sampled boundaries.
    """
    spans = req.metadata.get(IMAGE_SPANS_KEY)
    tokens = req.input_ids + list(accumulated_tokens)
    if not spans:
        compacted = compact_image_tokens(req) + list(accumulated_tokens)
        tokenizer = getattr(req.processor, "tokenizer", None)
        image_id = getattr(req.processor, "image_token_id", None)
        if (
            req.image_data
            and tokenizer is not None
            and tokenizer.convert_tokens_to_ids("<|image_pad|>") == image_id
            and compacted.count(image_id) != len(req.image_data)
        ):
            raise ValueError(
                "Ambiguous Qwen image tokens require explicit real-image spans; "
                "cannot infer images from sampled token IDs"
            )
        return {"input_ids": compacted}
    image_id = getattr(req.processor, "image_token_id", None)
    validate_image_spans(tokens, spans, image_id, len(req.image_data))
    tokenizer = req.processor.tokenizer
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    allowed = set()
    for start, end in spans:
        allowed.update(range(start, end + 1))
        if start > 0 and tokens[start - 1] == vision_start:
            allowed.add(start - 1)
        if end + 1 < len(tokens) and tokens[end + 1] == vision_end:
            allowed.add(end + 1)
    visual_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
    }
    if any(
        token in visual_ids and index not in allowed
        for index, token in enumerate(tokens)
    ):
        return {
            "input_ids": tokens,
            "mm_sampling_kwargs": {IMAGE_SPANS_KEY: spans},
        }
    return {"input_ids": compact_image_tokens(req) + list(accumulated_tokens)}


def validate_image_spans(
    tokens: list[int], spans: list[list[int]], image_id: int, image_count: int
) -> None:
    """Reject stale/ambiguous metadata before assigning any image features."""
    if not isinstance(spans, list) or len(spans) != image_count:
        raise ValueError("Real image spans must match image count")
    previous_end = -1
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("An image span must contain inclusive start and end")
        start, end = span
        if (
            type(start) is not int
            or type(end) is not int
            or not previous_end < start <= end < len(tokens)
        ):
            raise ValueError(
                "Image spans must be ordered, disjoint and within input_ids"
            )
        if any(token != image_id for token in tokens[start : end + 1]):
            raise ValueError("An image span contains non-image token IDs")
        previous_end = end


def compact_image_tokens(req: ModelRequest) -> list[int]:
    """Send one placeholder per image, retaining expanded IDs for training.

    SGLang expands image placeholders itself. Sending expanded runs can force
    it to decode and retokenize the prompt, changing sampled token boundaries.
    Only compact when the runs match the supplied images, preserving unknown
    processor formats. Always return a copy, including for text-only inputs.
    """
    image_token_id = getattr(req.processor, "image_token_id", None)
    if not req.image_data or not isinstance(image_token_id, int):
        return req.input_ids.copy()

    spans = req.metadata.get(IMAGE_SPANS_KEY)
    if spans:
        validate_image_spans(req.input_ids, spans, image_token_id, len(req.image_data))
        compact_ids = []
        cursor = 0
        for start, end in spans:
            compact_ids.extend(req.input_ids[cursor : start + 1])
            cursor = end + 1
        compact_ids.extend(req.input_ids[cursor:])
        return compact_ids

    compact_ids: list[int] = []
    image_count = 0
    previous_is_image = False
    for token in req.input_ids:
        is_image = token == image_token_id
        if is_image and previous_is_image:
            continue
        if is_image:
            image_count += 1
        compact_ids.append(token)
        previous_is_image = is_image

    if image_count != len(req.image_data):
        return req.input_ids.copy()
    return compact_ids
