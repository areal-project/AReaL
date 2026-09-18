# SPDX-License-Identifier: Apache-2.0

"""Sample reservations, owned by the dispatcher's input condition lock."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class SampleReservation:
    attempt_id: str
    group_size: int
    completed: set[int] = field(default_factory=set)


class SampleCapacity:
    """Reserve whole groups and release individual episodes idempotently.

    Callers must serialize all access. A reservation survives a remote timeout:
    only worker progress or confirmed execution termination releases its slots.
    """

    def __init__(self, limit: int):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_concurrent_samples must be a positive integer")
        self.limit = limit
        self.running = 0
        self.reservations: dict[int, SampleReservation] = {}

    def can_reserve(self, group_size: int) -> bool:
        return self.running + group_size <= self.limit

    def reserve(self, task_id: int, group_size: int) -> str:
        if task_id in self.reservations:
            raise ValueError(f"Task {task_id} still has a sample reservation")
        if not 1 <= group_size <= self.limit:
            raise ValueError("group_size must fit within max_concurrent_samples")
        if not self.can_reserve(group_size):
            raise RuntimeError("Insufficient sample capacity")
        attempt_id = uuid4().hex
        self.reservations[task_id] = SampleReservation(attempt_id, group_size)
        self.running += group_size
        return attempt_id

    def complete(self, task_id: int, attempt_id: str, sample_idx: int) -> bool:
        reservation = self.reservations.get(task_id)
        if (
            reservation is None
            or reservation.attempt_id != attempt_id
            or type(sample_idx) is not int
            or not 0 <= sample_idx < reservation.group_size
            or sample_idx in reservation.completed
        ):
            return False
        reservation.completed.add(sample_idx)
        self.running -= 1
        return True

    def finish(self, task_id: int, attempt_id: str) -> bool:
        """Release remaining slots only after execution is confirmed stopped."""
        reservation = self.reservations.get(task_id)
        if reservation is None or reservation.attempt_id != attempt_id:
            return False
        self.running -= reservation.group_size - len(reservation.completed)
        del self.reservations[task_id]
        return True
