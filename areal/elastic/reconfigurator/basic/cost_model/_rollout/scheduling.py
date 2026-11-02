# SPDX-License-Identifier: Apache-2.0

"""Explicit backend policies; these are simulation assumptions, not server traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from .types import ServiceSlice

if TYPE_CHECKING:
    from ...scenario import Request
    from ..rollout import RolloutConfig


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

    def choose_replica(
        self,
        config: RolloutConfig,
        request: RequestState,
        assigned: list[list[RequestState]],
        fixed_replica: int | None,
    ) -> int | None:
        need = request.request.input_tokens + request.request.output_tokens - 1
        candidates = [
            i
            for i, rows in enumerate(assigned)
            if (fixed_replica is None or i == fixed_replica)
            and len(rows) < config.max_batch_sequences
            and sum(r.request.input_tokens + r.request.output_tokens - 1 for r in rows)
            + need
            <= config.kv_capacity_tokens
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda i: (
                sum(
                    r.request.input_tokens + r.request.output_tokens
                    for r in assigned[i]
                ),
                i,
            ),
        )

    def batch(
        self, config: RolloutConfig, rows: list[RequestState]
    ) -> tuple[tuple[ServiceSlice, ...], tuple[tuple[str, str], ...]]:
        budget = config.max_batch_tokens
        slices, keys = [], []
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
        return tuple(slices), tuple(keys)


_SGLANG = SGLangScheduler()


def scheduler_for(backend: str) -> SGLangScheduler:
    if backend != "sglang":
        raise ValueError("No scheduling policy for this rollout backend")
    return _SGLANG
