# SPDX-License-Identifier: Apache-2.0

"""Opt-in Arena stream provenance for rejection-aware rollout sampling."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class ArenaStreamDataset:
    """Preserve a dataset's mixture while allowing same-stream replacements.

    ``stream_key`` identifies the source of each prompt. The original row order
    defines the base stream schedule, including any existing mixture weights.
    Training draws rows through independent per-stream cursors so replacements
    advance the affected source without consuming another source's base slot.
    Wrap only the training dataset; validation should use the original dataset.
    """

    def __init__(self, dataset: Any, *, stream_key: str) -> None:
        self.dataset = dataset
        self.stream_key = stream_key
        self.streams: list[str] = []
        self.indices: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            stream = dataset[index].get(stream_key)
            if not isinstance(stream, str) or not stream:
                raise ValueError(
                    f"Row {index} must have a non-empty string {stream_key!r}"
                )
            self.streams.append(stream)
            self.indices[stream].append(index)
        if not self.streams:
            raise ValueError("ArenaStreamDataset must contain at least one row")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]
