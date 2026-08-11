# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class MOPDPhase(str, Enum):
    ROLLOUT = "rollout"
    TEACHER = "teacher"
    DRAIN = "drain"
    TRAIN = "train"


class MOPDPhaseMachine:
    """Validate the exclusive GPU-owner transitions within one MOPD step."""

    _ALLOWED = {
        MOPDPhase.ROLLOUT: {MOPDPhase.TEACHER},
        MOPDPhase.TEACHER: {MOPDPhase.DRAIN},
        MOPDPhase.DRAIN: {MOPDPhase.TRAIN},
        MOPDPhase.TRAIN: {MOPDPhase.ROLLOUT},
    }

    def __init__(self) -> None:
        self._phase = MOPDPhase.ROLLOUT

    @property
    def phase(self) -> MOPDPhase:
        return self._phase

    def transition(self, target: MOPDPhase) -> None:
        if target not in self._ALLOWED[self._phase]:
            raise RuntimeError(
                f"Invalid MOPD phase transition {self._phase.value} -> {target.value}"
            )
        self._phase = target

    def abort_to_rollout(self) -> None:
        """Record successful rollback to rollout ownership after a failed step."""
        self._phase = MOPDPhase.ROLLOUT
