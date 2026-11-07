# SPDX-License-Identifier: Apache-2.0

"""Explicit backend schedulers; these are simulation assumptions, not server traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol
from zlib import crc32

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
    """Decode-first/full-reservation prior, with optional internal Attention DP."""

    version: ClassVar[str] = "decode_first_full_reservation_v1"

    @staticmethod
    def attention_group(identity, dp_size):
        """Stable uniform routing prior, independent of process hash seeds."""
        return crc32(identity.encode()) % dp_size

    def choose_replica(self, config, request, replicas):
        """Choose a feasible replica using current KV load and stable tie breaks."""
        need = request.request.input_tokens + request.request.output_tokens - 1
        # Basic's frozen DTO predates internal DP; its degree remains one.
        dp_size = getattr(config, "dp_size", 1)
        inner = self.attention_group(
            f"{request.group_id}/{request.request.name}", dp_size
        )

        def fits(group):
            rows = [
                r
                for r in group.rows.values()
                if self.attention_group(f"{r.group_id}/{r.request.name}", dp_size)
                == inner
            ]
            return (
                len(rows) < config.max_batch_sequences // dp_size
                and sum(
                    r.request.input_tokens + r.request.output_tokens - 1 for r in rows
                )
                + need
                <= config.kv_capacity_tokens // dp_size
            )

        candidates = [
            identity
            for identity, group in replicas.items()
            if len(group.rows) < config.max_batch_sequences
            and group.reserved_tokens + need <= config.kv_capacity_tokens
            and (dp_size == 1 or fits(group))
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
        dp_size = getattr(config, "dp_size", 1)
        chunk = getattr(config, "prefill_chunk_tokens", None)
        prefill_budgets = (
            [min(budget, chunk // dp_size)] * dp_size if chunk is not None else None
        )
        separate = getattr(config, "pp", 1) > 1
        decode_only = separate and any(r.generated for r in rows)
        slices, keys = ([], [])
        for request in sorted(rows, key=lambda r: (not bool(r.generated), r.order)):
            if separate and bool(request.generated) != decode_only:
                continue
            if prefill_budgets is None and not budget:
                break
            identity = f"{request.group_id}/{request.request.name}"
            inner = self.attention_group(identity, dp_size)
            if request.generated:
                item = ServiceSlice(
                    identity,
                    request.kv_tokens,
                    1,
                    "decode",
                    True,
                )
            else:
                remaining = request.request.input_tokens - request.prefilled
                available = (
                    prefill_budgets[inner] if prefill_budgets is not None else budget
                )
                query = min(available, remaining)
                if not query:
                    continue
                item = ServiceSlice(
                    identity,
                    request.prefilled,
                    query,
                    "prefill",
                    query == remaining,
                )
            slices.append(item)
            keys.append((request.group_id, request.request.name))
            if prefill_budgets is None:
                budget -= item.query_tokens
            elif not request.generated:
                prefill_budgets[inner] -= item.query_tokens
        return (tuple(slices), tuple(keys))


_SGLANG = SGLangScheduler()


def scheduler_for(backend: str) -> SGLangScheduler:
    if backend != "sglang":
        raise ValueError("No scheduler for this rollout backend")
    return _SGLANG
