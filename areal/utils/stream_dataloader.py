# SPDX-License-Identifier: Apache-2.0

"""Dispatch-only replacement sampling; training results remain first-come-first-served."""

from __future__ import annotations

import random
from collections import deque
from threading import Lock
from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.dataset.stream import StreamDataset
from areal.utils.data import cycle_dataloader


class StreamDataLoader:
    """A checkpointable prompt source for ``RolloutController.prepare_batch``.

    The base loader yields stream names in the original dataset's sampling order.
    Rejected groups create debt in their source. A queued/running replacement
    reserves one debt until it succeeds; failure releases that reservation.
    No accepted results are buffered here and there are no per-batch quotas.
    After at most one batch's worth of consecutive replacements, a base draw
    keeps healthy sources reachable even if another source always rejects.

    Each stream cycles independently and optionally reshuffles when exhausted.
    Base and replacement draws share the same cursor. The nominal epoch length
    and base stream proportions are unchanged by replacement sampling.
    """

    def __init__(
        self,
        dataset: StreamDataset,
        base_loader: StatefulDataLoader,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        if len(base_loader) == 0:
            raise ValueError("StreamDataLoader requires at least one base batch")
        self.dataset = dataset
        self.batch_size = base_loader.batch_size
        self.sampler = base_loader.sampler
        self._base_loader = base_loader
        self._base_iterator = None
        self._pending_streams: deque[str] = deque()
        self._shuffle = shuffle
        self._seed = seed
        self._lock = Lock()
        self._epochs = dict.fromkeys(dataset.indices, 0)
        self._positions = dict.fromkeys(dataset.indices, 0)
        self._orders: dict[str, list[int]] = {}
        self._debt = dict.fromkeys(dataset.indices, 0)
        self._reserved = dict.fromkeys(dataset.indices, 0)
        self._tasks: dict[int, tuple[str, bool]] = {}
        self._replacement_streak = 0

    def __len__(self) -> int:
        return len(self._base_loader)

    def __iter__(self):
        raise RuntimeError(
            "StreamDataLoader requires RolloutController.prepare_batch for "
            "rejection feedback; use the unwrapped dataset for other consumers"
        )

    def _next_row(self, stream: str) -> dict[str, Any]:
        if self._positions[stream] == len(self.dataset.indices[stream]):
            self._epochs[stream] += 1
            self._positions[stream] = 0
            self._orders.pop(stream, None)
        if stream not in self._orders:
            order = list(self.dataset.indices[stream])
            if self._shuffle:
                random.Random(f"{self._seed}:{stream}:{self._epochs[stream]}").shuffle(
                    order
                )
            self._orders[stream] = order
        index = self._orders[stream][self._positions[stream]]
        row = dict(self.dataset[index])
        self._positions[stream] += 1
        return row

    def next_rollout(self, task_id: int) -> dict[str, Any]:
        """Select and reserve a source before the task is enqueued."""
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Rollout task {task_id} is already reserved")
            stream = max(
                self._debt, key=lambda name: self._debt[name] - self._reserved[name]
            )
            replacement = self._debt[stream] > self._reserved[
                stream
            ] and self._replacement_streak < (self.batch_size or 1)
            if not replacement:
                if not self._pending_streams:
                    if self._base_iterator is None:
                        self._base_iterator = cycle_dataloader(self._base_loader)
                    self._pending_streams.extend(next(self._base_iterator))
                stream = self._pending_streams[0]
            row = self._next_row(stream)
            if replacement:
                self._reserved[stream] += 1
                self._replacement_streak += 1
            else:
                self._pending_streams.popleft()
                self._replacement_streak = 0
            self._tasks[task_id] = (stream, replacement)
            return row

    def complete_rollout(self, task_id: int, accepted: bool | None) -> None:
        """Account once for completion, or release a cancelled reservation.

        ``None`` denotes cancellation before a result was observed, not a
        confirmed rejection. Repeated notifications are harmless.
        """
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task is None:
                return
            stream, replacement = task
            if replacement:
                self._reserved[stream] -= 1
                if accepted:
                    self._debt[stream] -= 1
            elif accepted is False:
                self._debt[stream] += 1

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "base": self._base_loader.state_dict(),
                "pending_streams": list(self._pending_streams),
                "stream_key": self.dataset.stream_key,
                "stream_sizes": {
                    stream: len(indices)
                    for stream, indices in self.dataset.indices.items()
                },
                "shuffle": self._shuffle,
                "seed": self._seed,
                "epochs": self._epochs.copy(),
                "positions": self._positions.copy(),
                # Include reserved debt. Queued/running tasks do not survive
                # recovery, so their replacements must be eligible again.
                "debt": self._debt.copy(),
                "replacement_streak": self._replacement_streak,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            if self._tasks:
                raise RuntimeError("Cannot restore a stream loader with active tasks")
            if "stream_key" not in state:
                raise ValueError(
                    "Cannot restore a flat dataloader checkpoint into StreamDataLoader: "
                    "per-stream cursors and replacement debt are missing. Start a new "
                    "run or resume using the original unwrapped dataset."
                )
            sizes = {
                stream: len(indices) for stream, indices in self.dataset.indices.items()
            }
            if (
                state.get("stream_key") != self.dataset.stream_key
                or state.get("stream_sizes") != sizes
                or state.get("shuffle") != self._shuffle
                or state.get("seed") != self._seed
            ):
                raise ValueError(
                    "Stream sampling checkpoint does not match the dataset"
                )
            self._base_loader.load_state_dict(state["base"])
            self._pending_streams = deque(state["pending_streams"])
            self._epochs = state["epochs"].copy()
            self._positions = state["positions"].copy()
            self._debt = state["debt"].copy()
            self._replacement_streak = state["replacement_streak"]
            self._reserved = dict.fromkeys(self._debt, 0)
            self._orders.clear()
            self._base_iterator = None
