# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatTemplatePatch:
    """Exact string replacement for a known model chat template variant."""

    name: str
    search: str
    replacement: str


_KEEP_ALL_REASONING_PATCHES: Sequence[ChatTemplatePatch] = (
    ChatTemplatePatch(
        name="minimax_keep_all_reasoning",
        search="{%- if reasoning_content and loop.index0 > ns.last_user_index -%}",
        replacement="{%- if reasoning_content -%}",
    ),
)


def apply_keep_all_reasoning_patches(tokenizer: Any) -> list[str]:
    """Patch known templates to keep reasoning from all assistant turns.

    Patches are applied only to the tokenizer instance owned by the current
    process; no chat template file on disk is modified.

    Returns
    -------
    list[str]
        Names of patches that matched and were applied.
    """

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        return []

    applied: list[str] = []
    for patch in _KEEP_ALL_REASONING_PATCHES:
        if patch.search not in chat_template:
            continue
        chat_template = chat_template.replace(patch.search, patch.replacement, 1)
        applied.append(patch.name)

    if applied:
        tokenizer.chat_template = chat_template
    return applied
