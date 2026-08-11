# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    TaskIdGenerator,
    WorkflowExecutor,
)
from areal.utils.data import cycle_dataloader


class _VersionProvider:
    def get_version(self) -> int:
        return 0


@dataclass
class _TaskInput:
    task_id: int
    value: int
    accepted: bool = True


def _make_dispatcher() -> BatchTaskDispatcher[_TaskInput, int]:
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=4,
        consumer_batch_size=4,
        max_staleness=10,
    )

    def task_factory(task_input: _TaskInput):
        async def run() -> int | None:
            await asyncio.sleep(0.001)
            if not task_input.accepted:
                manager.on_rollout_rejected()
                return None
            manager.on_rollout_accepted()
            return task_input.value

        return run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=8,
        task_factory=task_factory,
        staleness_manager=manager,
    )
    dispatcher.initialize(logging.getLogger("WorkflowExecutorRecoverTest"))
    return dispatcher


def test_default_prepare_batch_keeps_async_readahead_path(monkeypatch):
    """The existing async dispatcher remains the default without the label."""
    monkeypatch.delenv("AREAL_DETERMINISTIC_SAMPLING", raising=False)
    executor = object.__new__(WorkflowExecutor)
    executor._dispatcher = MagicMock()
    executor._dispatcher.active_submit_and_wait.return_value = []
    executor.data_generator = iter(())

    executor.prepare_batch(
        SimpleNamespace(batch_size=4),
        workflow=MagicMock(),
    )

    executor._dispatcher.active_submit_and_wait.assert_called_once()
    executor._dispatcher.checkpoint_safe_submit_and_wait.assert_not_called()


def test_deterministic_prepare_batch_uses_checkpoint_safe_path(monkeypatch):
    """The no-read-ahead path requires the explicit deterministic label."""
    monkeypatch.setenv("AREAL_DETERMINISTIC_SAMPLING", "1")
    executor = object.__new__(WorkflowExecutor)
    executor._dispatcher = MagicMock()
    executor._dispatcher.checkpoint_safe_submit_and_wait.return_value = []
    executor.data_generator = iter(())

    executor.prepare_batch(
        SimpleNamespace(batch_size=4),
        workflow=MagicMock(),
    )

    executor._dispatcher.checkpoint_safe_submit_and_wait.assert_called_once()
    executor._dispatcher.active_submit_and_wait.assert_not_called()


def test_checkpoint_safe_batch_does_not_consume_next_batch():
    """Checkpoint-safe dispatch stops reading after the current batch."""
    dispatcher = _make_dispatcher()
    consumed: list[int] = []

    def inputs():
        for task_id in range(8):
            consumed.append(task_id)
            yield _TaskInput(task_id=task_id, value=task_id)

    try:
        results = dispatcher.checkpoint_safe_submit_and_wait(inputs(), batch_size=4)
        assert sorted(results) == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]
    finally:
        dispatcher.destroy()


def test_checkpoint_safe_batch_rejection_fails_closed():
    """A rejected task does not consume an uncheckpointed replacement input."""
    dispatcher = _make_dispatcher()
    consumed: list[int] = []

    def inputs():
        for task_id in range(8):
            consumed.append(task_id)
            yield _TaskInput(
                task_id=task_id,
                value=task_id,
                accepted=task_id != 2,
            )

    try:
        with pytest.raises(RuntimeError, match="rejected 1 rollout"):
            dispatcher.checkpoint_safe_submit_and_wait(inputs(), batch_size=4)
        assert consumed == [0, 1, 2, 3]
    finally:
        dispatcher.destroy()


def test_task_id_generator_state_round_trip_continues_sequence():
    """Recovered task IDs continue after the last submitted task."""
    generator = TaskIdGenerator()
    assert [generator.next() for _ in range(3)] == [0, 1, 2]

    recovered = TaskIdGenerator()
    recovered.load_state_dict(generator.state_dict())

    assert recovered.next() == 3


@pytest.mark.parametrize("next_task_id", [-1, True, 1.5, "3"])
def test_task_id_generator_rejects_invalid_recover_state(next_task_id):
    """Invalid counters fail instead of silently changing sampling identity."""
    generator = TaskIdGenerator()
    with pytest.raises(ValueError, match="non-negative integer"):
        generator.load_state_dict({"next_task_id": next_task_id})


def test_dataloader_and_task_ids_resume_at_same_batch_boundary():
    """Fresh recovery produces the same next batch and task IDs."""

    def inputs(dataloader, task_ids):
        for batch in cycle_dataloader(dataloader):
            for value in batch:
                yield _TaskInput(
                    task_id=task_ids.next(),
                    value=int(value),
                )

    dataloader = StatefulDataLoader(range(40), batch_size=4, shuffle=False)
    task_ids = TaskIdGenerator()
    data_generator = inputs(dataloader, task_ids)
    dispatcher = _make_dispatcher()
    try:
        for _ in range(3):
            dispatcher.checkpoint_safe_submit_and_wait(
                data_generator,
                batch_size=4,
            )
        dataloader_state = copy.deepcopy(dataloader.state_dict())
        task_id_state = task_ids.state_dict()
        continuous_next = dispatcher.checkpoint_safe_submit_and_wait(
            data_generator,
            batch_size=4,
        )
    finally:
        dispatcher.destroy()

    recovered_dataloader = StatefulDataLoader(range(40), batch_size=4, shuffle=False)
    recovered_dataloader.load_state_dict(dataloader_state)
    recovered_task_ids = TaskIdGenerator()
    recovered_task_ids.load_state_dict(task_id_state)
    recovered_dispatcher = _make_dispatcher()
    try:
        recovered_next = recovered_dispatcher.checkpoint_safe_submit_and_wait(
            inputs(recovered_dataloader, recovered_task_ids),
            batch_size=4,
        )
    finally:
        recovered_dispatcher.destroy()

    assert sorted(continuous_next) == [12, 13, 14, 15]
    assert sorted(recovered_next) == sorted(continuous_next)
    assert recovered_task_ids.state_dict() == {"next_task_id": 16}
