# SPDX-License-Identifier: Apache-2.0

"""Immutable input templates shared by counterfactuals of one observation."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from areal.elastic.policy.basic.forecast import (
    EpisodeSample,
    Forecast,
    OnlineObservation,
    RequestView,
)

from .online_state import _REQUEST_FIELDS


@dataclass(frozen=True)
class WorkInputs:
    episodes: tuple[EpisodeSample, ...]
    truth: Mapping[str, EpisodeSample]
    heap: tuple[tuple, ...]
    sequence: int

    @classmethod
    def prepare(cls, episodes, closes_at, start, known=frozenset()):
        truth = {e.episode_id: e for e in episodes}
        if len(truth) != len(episodes):
            raise ValueError("Duplicate episode identity")
        if any(e.available_at > closes_at for e in episodes):
            raise ValueError("Input closes before an arrival")
        if closes_at < start - 1e-9:
            raise ValueError("An event cannot be scheduled in the past")
        events, sequence = [], 0
        for episode in episodes:
            if episode.available_at >= start:
                sequence += 1
                if episode.episode_id not in known:
                    heapq.heappush(
                        events,
                        (
                            episode.available_at,
                            1,
                            sequence,
                            "arrival",
                            episode.episode_id,
                        ),
                    )
        sequence += 1
        heapq.heappush(
            events, (max(start, closes_at), 1, sequence, "input_closed", None)
        )
        return cls(episodes, MappingProxyType(truth), tuple(events), sequence)


@dataclass(frozen=True)
class ForecastInputs:
    view: OnlineObservation
    forecast: Forecast
    work: WorkInputs
    requests: tuple[tuple[RequestView, int, tuple], ...]
    order: tuple[str, ...]
    active: int
    completed: int
    admission_cursor: int

    @classmethod
    def prepare(cls, view, forecast, max_new_tokens, max_context_tokens):
        order = tuple(e.episode_id for e in view.episodes)
        work = WorkInputs.prepare(
            forecast.episodes, forecast.input_closes_at, view.now, frozenset(order)
        )
        requests = []
        for request in view.requests:
            current = replace(
                request,
                deployment=1
                if request.deployment == view.rollout_deployment
                else request.deployment,
            )
            reservation = min(max_new_tokens, max_context_tokens - current.input_tokens)
            requests.append(
                (
                    current,
                    reservation,
                    tuple(getattr(current, n) for n in _REQUEST_FIELDS),
                )
            )
        return cls(
            view,
            forecast,
            work,
            tuple(requests),
            order,
            sum(e.admitted and e.completed_at is None for e in view.episodes),
            sum(e.completed_at is not None for e in view.episodes),
            next(
                (
                    i
                    for i, e in enumerate(view.episodes)
                    if not e.admitted and e.completed_at is None
                ),
                len(order),
            ),
        )
