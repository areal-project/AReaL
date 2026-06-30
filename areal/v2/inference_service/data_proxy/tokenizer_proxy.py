# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from areal.utils.hf_utils import apply_chat_template as _apply_chat_template


class TokenizerProxy:
    """Wraps HuggingFace tokenizer with async-safe methods for the data proxy."""

    def __init__(self, tokenizer_path: str):
        from areal.utils.hf_utils import load_hf_tokenizer

        self._tok = load_hf_tokenizer(tokenizer_path)

    async def tokenize(self, text: str) -> list[int]:
        """Tokenize string -> token IDs. Runs in executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._tok.encode, text)

    async def apply_chat_template(self, messages: list[dict], **kw) -> list[int]:
        """Apply chat template -> token IDs. Runs in executor."""
        loop = asyncio.get_running_loop()

        def _apply():
            return _apply_chat_template(
                self._tok, messages, tokenize=True, add_generation_prompt=True, **kw
            )

        return await loop.run_in_executor(None, _apply)

    def decode_token(self, token_id: int) -> str:
        """Decode single token ID -> string piece. Sync (fast dict lookup)."""
        return self._tok.decode([token_id], skip_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs -> full string. Used by ChatCompletionHandler."""
        return self._tok.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id or self._tok.eos_token_id
