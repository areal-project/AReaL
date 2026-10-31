# SPDX-License-Identifier: Apache-2.0

"""Finite ordered task supply; policies receive metadata without future outputs."""

from dataclasses import dataclass

from areal.elastic.policy.basic.observations import EpisodeSample


@dataclass(frozen=True)
class PendingTask:
    episode_id: str
    family: str
    input_tokens: int


class TaskPool:
    def __init__(self, episodes: tuple[EpisodeSample, ...]):
        self._tasks = episodes
        self.cursor = 0
        if len({e.episode_id for e in episodes}) != len(episodes):
            raise ValueError("Task identities must be unique")

    def __len__(self):
        return len(self._tasks)

    def pending(self) -> tuple[PendingTask, ...]:
        return tuple(
            PendingTask(e.episode_id, e.family, e.turns[0].input_tokens)
            for e in self._tasks[self.cursor :]
        )

    def take(self) -> EpisodeSample:
        task = self._tasks[self.cursor]
        self.cursor += 1
        return task
