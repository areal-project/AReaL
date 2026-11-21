# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch rollout/backward overlap with a single global commit."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.cli_args import StreamTrainerConfig
    from areal.infra.controller.rollout_controller import RolloutController
    from areal.infra.controller.train_controller import TrainController

logger = logging.getLogger("StreamTrainer")


@dataclass(frozen=True)
class _ActorReplica:
    actor_ranks: list[int]
    rollout_ranks: list[int]


class StreamTrainer:
    """Move only drained, physically matched replicas from inference to backward.

    The ordinary collector still owns batch membership and filtering. This
    coordinator sees only groups it actually consumes, and verifies that exact
    membership again before allowing an optimizer step. It never changes the
    training world or publishes partial weights.
    """

    def __init__(
        self,
        actor: TrainController,
        rollout: RolloutController,
        config: StreamTrainerConfig,
    ) -> None:
        self.actor = actor
        self.rollout = rollout
        self.config = config
        self._replicas: dict[int, _ActorReplica] = {}
        self._server_replicas: dict[int, int] = {}
        self._update_id: str | None = None
        self._started_at = 0.0
        self._collected = False

    @staticmethod
    def _gpu_key(host: str, gpu_uuid: str) -> tuple[str, str]:
        normalized = gpu_uuid.lower().removeprefix("gpu-")
        if not normalized:
            raise RuntimeError("Stream Trainer requires nonempty physical GPU UUIDs")
        return host, normalized

    def _discover_replicas(self) -> None:
        actors = self.actor.stream_topology()
        servers = self.rollout.stream_topology()
        actor_groups: dict[int, list[dict[str, Any]]] = {}
        for rank in actors:
            actor_groups.setdefault(rank["dp_rank"], []).append(rank)
        groups_by_gpus: dict[frozenset[tuple[str, str]], tuple[int, list[int]]] = {}
        all_actor_gpus: set[tuple[str, str]] = set()
        for dp_rank, members in actor_groups.items():
            ranks = sorted(member["rank"] for member in members)
            if any(sorted(member["replica_ranks"]) != ranks for member in members):
                raise RuntimeError(
                    "Actor ranks disagree about their complete TP replica"
                )
            if sum(bool(self.actor.workers_is_dp_head[rank]) for rank in ranks) != 1:
                raise RuntimeError("A streamed actor replica must have one DP head")
            gpus = frozenset(
                self._gpu_key(member["host"], member["gpu_uuid"]) for member in members
            )
            if len(gpus) != len(ranks) or all_actor_gpus.intersection(gpus):
                raise RuntimeError("Actor replicas overlap physical GPUs")
            all_actor_gpus.update(gpus)
            groups_by_gpus[gpus] = dp_rank, ranks
        servers_by_actor: dict[int, list[int]] = {
            dp_rank: [] for dp_rank in actor_groups
        }
        server_replicas = {}
        all_server_gpus: set[tuple[str, str]] = set()
        for server in servers:
            worker_rank = server["worker_rank"]
            if (
                not isinstance(worker_rank, int)
                or not 0 <= worker_rank < len(self.rollout.workers)
                or worker_rank in server_replicas
            ):
                raise RuntimeError("Invalid SGLang worker in streamed topology")
            host = self.rollout.workers[worker_rank].ip
            gpus = frozenset(
                self._gpu_key(host, value) for value in server["gpu_uuids"]
            )
            if (
                not gpus
                or len(gpus) != len(server["gpu_uuids"])
                or all_server_gpus.intersection(gpus)
            ):
                raise RuntimeError("SGLang replicas overlap physical GPUs")
            matches = [
                dp_rank
                for actor_gpus, (dp_rank, _) in groups_by_gpus.items()
                if gpus <= actor_gpus
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Every SGLang replica must fit inside one complete actor TP replica"
                )
            dp_rank = matches[0]
            servers_by_actor[dp_rank].append(worker_rank)
            server_replicas[worker_rank] = dp_rank
            all_server_gpus.update(gpus)
        if all_server_gpus != all_actor_gpus or set(server_replicas) != set(
            range(len(self.rollout.workers))
        ):
            raise RuntimeError(
                "SGLang replicas must partition every actor replica's physical GPUs"
            )
        self._replicas = {
            dp_rank: _ActorReplica(ranks, sorted(servers_by_actor[dp_rank]))
            for dp_rank, ranks in groups_by_gpus.values()
        }
        self._server_replicas = server_replicas

    def _event(self, name: str, **fields: Any) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(
            "[StreamTrainer] event=%s update_id=%s elapsed_s=%.6f %s",
            name,
            self._update_id,
            time.monotonic() - self._started_at,
            details,
        )

    def _backward(self, event: dict[str, Any], dp_rank: int) -> float:
        ranks = self._replicas[dp_rank].actor_ranks
        chunk_id = event["task_id"]
        self._event("backward_start", chunk_id=chunk_id, dp_rank=dp_rank)
        self.actor.stream_accumulate(
            self._update_id,
            chunk_id,
            [event["trajectory"]],
            worker_ranks=ranks,
        )
        completed_at = time.monotonic()
        self._event("backward_done", chunk_id=chunk_id, dp_rank=dp_rank)
        return completed_at

    def collect_and_accumulate(
        self,
        prepare_batch: Callable[[], list[dict[str, Any]]],
        *,
        global_step: int,
    ) -> list[dict[str, Any]]:
        if self._update_id is not None:
            raise RuntimeError("Commit or abort the previous streamed batch first")
        if not self._replicas:
            self._discover_replicas()
        self._update_id = f"step-{global_step}-v{self.rollout.get_version()}"
        self._started_at = time.monotonic()
        self._collected = False
        pending: deque[dict[str, Any]] = deque()
        observed: dict[int, int] = {}
        eligible: set[int] = set()
        drained: set[int] = set()
        released: set[int] = set()
        jobs: dict[int, Future[float]] = {}
        completed: list[tuple[float, int]] = []
        collection_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stream-collect"
        )
        backward_pool = ThreadPoolExecutor(
            max_workers=len(self._replicas), thread_name_prefix="stream-backward"
        )
        collection_started = False

        def collect() -> tuple[list[dict[str, Any]], float]:
            batch = prepare_batch()
            finished_at = time.monotonic()
            self._event("rollout_complete", groups=len(batch))
            return batch, finished_at

        def consume_events(timeout: float) -> None:
            for event in self.rollout.get_stream_completed(timeout=timeout):
                chunk_id = event["task_id"]
                if chunk_id in observed:
                    raise RuntimeError("A rollout group was consumed more than once")
                observed[chunk_id] = id(event["trajectory"])
                eligible.add(event["worker_rank"])
                pending.append(event)

        def reap() -> None:
            for rank, job in list(jobs.items()):
                if job.done():
                    completed.append((job.result(), rank))
                    del jobs[rank]

        def schedule() -> None:
            for rank in sorted(drained):
                if pending and rank not in jobs:
                    jobs[rank] = backward_pool.submit(
                        self._backward, pending.popleft(), rank
                    )

        try:
            self.rollout.begin_stream_collection()
            collection_started = True
            # The dispatcher may already hold prefetched inputs from the last
            # step. Observe their worker reservations before admitting them.
            self.rollout.resume()
            future = collection_pool.submit(collect)
            while not future.done():
                consume_events(self.config.poll_interval)
                reap()
                if pending:
                    # Only consider replicas that have completed a group; do
                    # not steal an unused replica before initial submission.
                    for dp_rank, replica in sorted(self._replicas.items()):
                        if (
                            dp_rank in drained
                            or not set(replica.rollout_ranks) <= eligible
                        ):
                            continue
                        if self.rollout.try_drain_stream_replicas(
                            replica.rollout_ranks, keep_one=True
                        ):
                            drained.add(dp_rank)
                            for rank in replica.rollout_ranks:
                                released.add(rank)
                                self._event(
                                    "replica_drained",
                                    worker_rank=rank,
                                    dp_rank=dp_rank,
                                )
                            self._event("actor_replica_ready", dp_rank=dp_rank)
                schedule()

            batch, rollout_finished_at = future.result()
            consume_events(0.0)
            if len(observed) != len(batch) or set(observed.values()) != {
                id(group) for group in batch
            }:
                raise RuntimeError(
                    "Streamed groups do not exactly match the collected training batch"
                )
            # No running workflow may remain at this fixed-batch boundary.
            # This also releases the last server exactly once, before any
            # actor operation on its physical GPUs.
            for rank in self.rollout.finish_stream_draining():
                if rank not in released:
                    self._event(
                        "replica_drained",
                        worker_rank=rank,
                        dp_rank=self._server_replicas[rank],
                    )
                released.add(rank)
            for dp_rank, replica in self._replicas.items():
                if not set(replica.rollout_ranks) <= released:
                    raise RuntimeError("Actor replica still shares live inference GPUs")
                if dp_rank not in drained:
                    drained.add(dp_rank)
                    self._event("actor_replica_ready", dp_rank=dp_rank)
            self.rollout.pause()
            while pending or jobs:
                reap()
                schedule()
                if jobs:
                    # Wait on one bounded RPC future at a time; RPC timeouts
                    # propagate and forbid a partial optimizer commit.
                    next(iter(jobs.values())).result()
            early = [
                (when, rank) for when, rank in completed if when < rollout_finished_at
            ]
            stats_tracker.scalar(
                **{
                    "stream_trainer/total_groups": len(batch),
                    "stream_trainer/early_groups": len(early),
                    "stream_trainer/early_replicas": len({rank for _, rank in early}),
                    "stream_trainer/rollout_seconds": rollout_finished_at
                    - self._started_at,
                    "stream_trainer/collection_and_backward_seconds": time.monotonic()
                    - self._started_at,
                }
            )
            self._collected = True
            return batch
        except BaseException:
            # A backward can fail after modifying gradients. Never fall back to
            # ordinary ppo_update on those buffers, or reopen inference memory.
            self.rollout.pause()
            for job in jobs.values():
                job.cancel()
            backward_pool.shutdown(wait=True, cancel_futures=True)
            try:
                self.actor.abort_stream_update(self._update_id)
            except Exception:
                logger.exception("Failed to discard an incomplete streamed update")
            raise
        finally:
            collection_pool.shutdown(wait=False, cancel_futures=True)
            backward_pool.shutdown(wait=True, cancel_futures=True)
            if collection_started:
                self.rollout.end_stream_collection()

    def commit(self) -> None:
        if self._update_id is None or not self._collected:
            raise RuntimeError("A complete streamed batch is required before commit")
        self._event("optimizer_start")
        self.actor.finish_stream_update(self._update_id)
        self._event("optimizer_done")
        self._update_id = None
        self._collected = False

    def release_rollout(self) -> None:
        """Clear admission fences after the ordinary global AWEX handoff."""
        if self._update_id is not None:
            raise RuntimeError(
                "Cannot release inference before a streamed update commits"
            )
        self.rollout.restore_stream_replicas()
