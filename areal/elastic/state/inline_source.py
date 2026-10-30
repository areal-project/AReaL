# SPDX-License-Identifier: Apache-2.0

"""A CPU-owned producer around the native inline dataset/workflow dispatcher.

Only Actor workers restart. The dataloader, workflow executor, counters and
serving replicas stay alive; coordinator crash recovery is deliberately excluded.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any

from .batch_queue import BatchConsumer, BatchQueue


class InlineBatchSource:
    def __init__(
        self, root: Path, *, batch_size: int, max_staleness: int, group_size: int = 1
    ):
        self.queue = BatchQueue(root)
        # Native dispatch, admission credits and batches count input task groups.
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_staleness = max_staleness
        self.rollout = None
        self._thread = None
        self._closed = threading.Event()
        self._error = None
        self._published_version = 0
        self._admission_open = False
        self._lock = threading.RLock()

    def start(self, rollout, *, total_batches: int, **prepare_kwargs):
        if self._thread is not None:
            raise RuntimeError("Only one native inline producer may own the dataloader")
        self.rollout = rollout

        def produce():
            from areal.infra.rpc.rtensor import RTensor
            from areal.infra.rpc.serialization import serialize_value
            from areal.infra.utils.concurrent import run_async_task

            try:
                for ordinal in range(total_batches):
                    if self._closed.is_set():
                        return
                    started = time.monotonic()
                    trajectories = rollout.prepare_batch(**prepare_kwargs)
                    if self._closed.is_set():
                        return
                    if len(trajectories) != self.batch_size:
                        raise ValueError(
                            "Native inline batch has an unexpected task-group count"
                        )
                    shards = RTensor.collect_shards(trajectories)
                    trajectories = RTensor.localize(trajectories)
                    for trajectory in trajectories:
                        self._inspect_payload(trajectory)
                        self.inspect_group(trajectory)
                    self.queue.enqueue(
                        f"inline-{ordinal}",
                        {
                            "episode_ids": [
                                f"inline-{ordinal}-{i}" for i in range(self.batch_size)
                            ],
                            "group_size": self.group_size,
                            "trajectories": [serialize_value(t) for t in trajectories],
                            "collection_seconds": time.monotonic() - started,
                        },
                        policy_version=ordinal,
                    )
                    # The self-contained batch is durable before remote tensors are retired.
                    for node, shard_ids in shards.items():
                        run_async_task(RTensor.clear_node, node, shard_ids)
            except BaseException as error:
                if not self._closed.is_set():
                    self._error = error

        self._thread = threading.Thread(
            target=produce, name="inline-batches", daemon=True
        )
        self._thread.start()

    def seal_batch(self, actor_version: int, *, consumer: BatchConsumer) -> str | None:
        if self._error is not None:
            raise RuntimeError("Native inline producer failed") from self._error
        batch = consumer.peek()
        if batch is None:
            return None
        if batch.policy_version != actor_version:
            raise RuntimeError("Inline batch order differs from Actor update progress")
        return batch.batch_id

    def confirm_publication(self, version: int, *, consumer: BatchConsumer):
        if version != consumer.cursor["applied_batches"]:
            raise RuntimeError("Publication must match the applied native batch cursor")
        with self._lock:
            self._published_version = version

    def set_admission_open(self, enabled: bool):
        # Lifecycle fence only. Native StalenessManager owns actual admission.
        self._admission_open = enabled

    def rollback(self, consumer):
        raise RuntimeError("Coordinator rollback is not supported by the inline demo")

    def status(self) -> dict[str, Any]:
        queue = self.queue.status()
        with self._lock:
            published = self._published_version
        counts = queue["counts"]
        native = (
            {}
            if self.rollout is None
            else dataclasses.asdict(self.rollout.staleness_manager.get_stats())
        )
        reserved = counts["leased"] * self.batch_size
        ready = max(
            0, native.get("accepted", 0) - published * self.batch_size - reserved
        )
        return {
            "published_version": published,
            "count_unit": "task_groups",
            "group_size": self.group_size,
            "counts": {
                "ready": ready,
                "reserved": reserved,
                "published": published * self.batch_size,
                "admitted": native.get("running", 0),
            },
            "rollout": native,
            "source_running": self._thread is not None and self._thread.is_alive(),
        }

    def inspect_group(self, payload: dict[str, Any]) -> None:
        # A linear concat workflow exports one row per sample. Never repair a
        # malformed group by borrowing rows from another task: GRPO needs it whole.
        if self.group_size > 1 and payload["input_ids"].shape[0] != self.group_size:
            raise ValueError(
                f"Expected {self.group_size} concat trajectories per task group"
            )

    def close(self):
        self._closed.set()
        if self.rollout is not None:
            self.rollout.workflow_executor.dispatcher.request_shutdown()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Inline batch producer did not stop")

    @staticmethod
    def _inspect_payload(payload: dict[str, Any]) -> tuple[int, int]:
        import torch

        required = (
            "input_ids",
            "attention_mask",
            "loss_mask",
            "logprobs",
            "versions",
            "rewards",
        )
        if any(not isinstance(payload.get(key), torch.Tensor) for key in required):
            raise ValueError(
                "An inline trajectory must contain actual training tensors"
            )
        if any(
            value.device.type != "cpu"
            for value in payload.values()
            if isinstance(value, torch.Tensor)
        ):
            raise ValueError("Persistent trajectories must contain CPU tensors")
        shape = payload["input_ids"].shape
        if (
            len(shape) != 2
            or min(shape) < 1
            or any(payload[key].shape != shape for key in required[1:5])
        ):
            raise ValueError(
                "Inline token tensors must have matching nonempty [rows,tokens] shapes"
            )
        if (
            payload["rewards"].shape != (shape[0],)
            or not torch.isfinite(payload["rewards"]).all()
        ):
            raise ValueError("Each trajectory row requires a finite reward")
        versions = payload["versions"]
        if versions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Token versions must be integers")
        mask = payload["loss_mask"].bool() & payload["attention_mask"].bool()
        mask[:, 0] = False  # No next-token loss can train the first input token.
        effective = versions[mask]
        if not mask.any(dim=1).all() or (effective < 0).any():
            raise ValueError("Every effective generated loss token requires a version")
        if not torch.isfinite(payload["logprobs"][mask]).all():
            raise ValueError("Generated token logprobs must be finite")
        return int(effective.min()), int(effective.max())
