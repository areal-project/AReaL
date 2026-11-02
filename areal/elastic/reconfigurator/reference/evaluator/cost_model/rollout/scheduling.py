# SPDX-License-Identifier: Apache-2.0

"""Explicit backend schedulers; these are simulation assumptions, not server traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ServiceSlice,
)

if TYPE_CHECKING:
    from areal.elastic.reconfigurator.reference.evaluator.contracts import Request
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
        RolloutConfig,
    )


class RequestState(Protocol):
    request: Request
    group_id: str
    order: int
    prefilled: int
    generated: int

    @property
    def kv_tokens(self) -> int: ...


@dataclass(frozen=True)
class SGLangScheduler:
    """Legacy decode-first batching and full-length reservation, kept unchanged."""

    version: ClassVar[str] = "decode_first_full_reservation_v1"

    def choose_replica(self, config, request, replicas):
        """Choose a feasible replica using current KV load and stable tie breaks."""
        need = request.request.input_tokens + request.request.output_tokens - 1
        candidates = [
            identity
            for identity, group in replicas.items()
            if len(group.rows) < config.max_batch_sequences
            and group.reserved_tokens + need <= config.kv_capacity_tokens
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda identity: (
                replicas[identity].kv_tokens,
                len(replicas[identity].rows),
                identity,
            ),
        )

    def batch(
        self, config: RolloutConfig, rows: list[RequestState]
    ) -> tuple[tuple[ServiceSlice, ...], tuple[tuple[str, str], ...]]:
        budget = config.max_batch_tokens
        slices, keys = ([], [])
        for request in sorted(rows, key=lambda r: (not bool(r.generated), r.order)):
            if not budget:
                break
            if request.generated:
                item = ServiceSlice(
                    f"{request.group_id}/{request.request.name}",
                    request.kv_tokens,
                    1,
                    "decode",
                    True,
                )
            else:
                remaining = request.request.input_tokens - request.prefilled
                query = min(budget, remaining)
                item = ServiceSlice(
                    f"{request.group_id}/{request.request.name}",
                    request.prefilled,
                    query,
                    "prefill",
                    query == remaining,
                )
            slices.append(item)
            keys.append((request.group_id, request.request.name))
            budget -= item.query_tokens
        return (tuple(slices), tuple(keys))


_SGLANG = SGLangScheduler()


def scheduler_for(backend: str) -> SGLangScheduler:
    if backend != "sglang":
        raise ValueError("No scheduler for this rollout backend")
    return _SGLANG
