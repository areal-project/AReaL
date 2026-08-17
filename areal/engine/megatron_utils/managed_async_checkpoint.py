# SPDX-License-Identifier: Apache-2.0

"""Operation-local state for managed Megatron asynchronous checkpoints.

Megatron-Core 0.17 exposes no public notification for when CPU source tensors
are no longer referenced by an async save.  AReaL therefore retains this
transaction, and the optimizer mutation fence attached to it, until the whole
``AsyncRequest`` has finalized in the foreground process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ManagedAsyncSaveState(Enum):
    """Mutation-fence lifecycle for one managed asynchronous save."""

    IDLE = auto()
    SAVE_STAGING = auto()
    SAVE_IN_FLIGHT = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class ManagedAsyncSaveTransaction:
    """Manager-owned authority and progress for one managed async request."""

    checkpoint_id: str
    path: str
    leaves: tuple[Any, ...]
    control_group: Any
    logical_call_id: int
    expected_call_idx: int
    marker_leaves: list[dict[str, Any]]
    marker_leaves_digest: str
    state: ManagedAsyncSaveState = ManagedAsyncSaveState.SAVE_STAGING
    request: Any | None = None
    call_idx: int | None = None
    completion_callbacks: list[Callable[[], None]] = field(default_factory=list)
    error: BaseException | None = None
    marker_created: bool = False
    marker_authority: Any | None = None
    marker_commit_decided: bool = False
    marker_committed: bool = False
    marker_cleanup_diagnostic: str | None = None
    worker_recovery: Any | None = None
    recovery_token: Any | None = None

    @property
    def terminal(self) -> bool:
        return self.state in (
            ManagedAsyncSaveState.COMPLETE,
            ManagedAsyncSaveState.FAILED,
        )
