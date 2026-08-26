# SPDX-License-Identifier: Apache-2.0

"""Hermetic stand-ins for a VLM tokenizer/processor pair.

The vision export path only depends on the *shape* of what a HuggingFace
processor returns (expanded ``input_ids``, ``mm_token_type_ids``,
``pixel_values``, ``image_grid_thw``), so the CPU tests drive it with these
stubs instead of downloading a real VLM. ``pixel_values`` is derived from image
content so that deduplication can be exercised meaningfully.
"""

import base64
import re
from io import BytesIO
from typing import Any

import torch
from PIL import Image

IMAGE_PLACEHOLDER = "<|image|>"
EOS_TOKEN = "<|eos|>"
EOS_ID = 999
IMAGE_PAD_ID = 900
VIDEO_PAD_ID = 901
PATCHES_PER_IMAGE = 4
PATCH_DIM = 8

# Reserved placeholders the client must never let through as generated tokens.
VISION_PAD_TOKEN_IDS = {"<|image_pad|>": IMAGE_PAD_ID, "<|video_pad|>": VIDEO_PAD_ID}

_SPECIAL_RE = re.compile(r"(<\|eos\|>|<\|image\|>)")


def encode_text(text: str) -> list[int]:
    """Deterministically encode text, keeping special markers as single ids."""
    ids: list[int] = []
    for chunk in _SPECIAL_RE.split(text):
        if chunk == EOS_TOKEN:
            ids.append(EOS_ID)
        elif chunk == IMAGE_PLACEHOLDER:
            ids.append(IMAGE_PAD_ID)
        else:
            ids.extend(ord(c) % 500 + 1 for c in chunk)
    return ids


class StubTokenizer:
    """Chat-template renderer with just enough surface for the client."""

    eos_token_id = EOS_ID
    pad_token_id = EOS_ID

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        tools: Any = None,
        **kwargs: Any,
    ) -> str | list[int] | dict[str, list[int]]:
        rendered = ""
        for message in messages:
            rendered += f"<|start|>{message.get('role', 'user')}\n"
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image":
                        rendered += IMAGE_PLACEHOLDER
                    elif part.get("type") == "image_url":
                        # The client always strips these before rendering; if one
                        # survives, make the mismatch visible instead of silent.
                        raise AssertionError(
                            "image_url part reached the chat template unstripped"
                        )
                    elif part.get("type") == "text":
                        rendered += part.get("text", "")
            elif content is not None:
                rendered += str(content)
            rendered += EOS_TOKEN
        if add_generation_prompt:
            rendered += "<|start|>assistant\n"
        if not tokenize:
            return rendered
        # Match the return shape of the installed transformers version, which
        # areal.utils.hf_utils.apply_chat_template normalises.
        from areal.utils import pkg_version

        ids = encode_text(rendered)
        if pkg_version.is_version_greater_or_equal("transformers", "5.0"):
            return {"input_ids": ids}
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return encode_text(text)

    def __call__(
        self, text: list[str], padding: bool = False, **kwargs: Any
    ) -> dict[str, list[list[int]]]:
        """Tokenize without expanding placeholders, as a processor delegate."""
        return {"input_ids": [encode_text(item) for item in text]}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return VISION_PAD_TOKEN_IDS.get(token)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        for token, tid in VISION_PAD_TOKEN_IDS.items():
            if tid == token_id:
                return token
        return None


class StubProcessor:
    """Expands image placeholders and emits content-derived pixel values."""

    def __init__(self) -> None:
        # Real processors expand placeholders in the text and then delegate to
        # this tokenizer, which is how the collapsed prompt is produced.
        self.tokenizer = StubTokenizer()

    def __call__(
        self,
        *,
        images: list[Image.Image],
        text: list[str],
        padding: bool = False,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        assert len(text) == 1, "stub processor handles one prompt at a time"
        parts = text[0].split(IMAGE_PLACEHOLDER)
        assert len(parts) - 1 == len(images), (
            f"prompt has {len(parts) - 1} image placeholders but "
            f"{len(images)} images were supplied"
        )

        input_ids: list[int] = []
        mm_token_type_ids: list[int] = []
        for index, part in enumerate(parts):
            text_ids = encode_text(part)
            input_ids.extend(text_ids)
            mm_token_type_ids.extend([0] * len(text_ids))
            if index < len(images):
                input_ids.extend([IMAGE_PAD_ID] * PATCHES_PER_IMAGE)
                mm_token_type_ids.extend([1] * PATCHES_PER_IMAGE)

        pixel_rows = []
        for image in images:
            # One deterministic row block per image, keyed on its content.
            marker = float(image.getpixel((0, 0))[0])
            pixel_rows.append(torch.full((PATCHES_PER_IMAGE, PATCH_DIM), marker))

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([mm_token_type_ids], dtype=torch.long),
            "pixel_values": (
                torch.cat(pixel_rows) if pixel_rows else torch.zeros(0, PATCH_DIM)
            ),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long),
        }


def make_image(marker: int, size: tuple[int, int] = (8, 8)) -> Image.Image:
    """Build a solid-colour image whose first pixel encodes ``marker``."""
    return Image.new("RGB", size, (marker, marker, marker))


def image_data_uri(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def user_message_with_image(image: Image.Image, text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
            {"type": "text", "text": text},
        ],
    }
