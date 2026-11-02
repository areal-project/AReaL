# SPDX-License-Identifier: Apache-2.0

"""Private experiment request, episode and deployment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from areal.elastic.contracts import (
    OnlineLayout,
    RequestView,
    ServiceView,
    ToolView,
    Turn,
)
from areal.elastic.reconfigurator.reference.evaluator.contracts import Request
from areal.elastic.simulation_primitives import ReplicaState, RequestState


@dataclass(init=False)
class _Request:
    view: RequestState
    reservation: int
    order: int

    def __init__(
        self,
        view: RequestView,
        reservation: int,
        order: int,
        *,
        values: tuple | None = None,
    ):
        self.view = (
            RequestState(*values) if values is not None else RequestState.copy(view)
        )
        self.reservation, self.order = (reservation, order)

    @property
    def group_id(self):
        return self.view.episode_id

    @cached_property
    def request(self):
        return Request(
            f"request-{self.view.turn}", self.view.input_tokens, self.reservation
        )

    @property
    def prefilled(self):
        return self.view.prefilled

    @property
    def generated(self):
        return self.view.generated

    @property
    def kv_tokens(self):
        return self.view.prefilled + max(0, self.view.generated - 1)


@dataclass
class _Episode:
    episode_id: str
    admitted: bool = False
    completed_at: float | None = None
    past: list[Turn] = field(default_factory=list)
    current: str | None = None
    tool: ToolView | None = None


@dataclass
class _Deployment:
    identity: int
    role: str
    layout: OnlineLayout
    status: str
    version: int
    busy: dict[int, ServiceView] = field(default_factory=dict)
    transfers: int = 0
    replicas: dict[int, ReplicaState] = field(default_factory=dict)

    def __post_init__(self):
        if self.role == "rollout":
            self.replicas = {
                i: ReplicaState() for i in range(self.layout.rollout_replicas)
            }
