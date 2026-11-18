# SPDX-License-Identifier: Apache-2.0

"""Public interfaces for optional worker-side data hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class HookConfig:
    """Import path and constructor keyword arguments for a hook class."""

    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RolloutHook(Protocol):
    """Transform workflow input and its exported tensor trajectory."""

    def pre_rollout(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, Any]: ...

    def post_rollout(
        self, trajectory: dict[str, Any], state: Any
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class ComputeLogpHook(Protocol):
    """Transform scoring inputs and align log-probabilities on output."""

    def pre_compute_logp(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], Any]: ...

    def post_compute_logp(
        self, outputs: list[torch.Tensor] | None, state: Any
    ) -> list[torch.Tensor] | None: ...
