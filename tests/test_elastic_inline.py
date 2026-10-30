# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of native inline collection and normal shutdown."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.test_elastic_runtime import _inline_payload

from areal.elastic.state.inline_source import InlineBatchSource
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher


def test_native_dispatcher_collects_without_actor_and_preserves_capacity(tmp_path):
    """A real dispatcher/staleness manager produces while no Actor exists."""
    version = SimpleNamespace(get_version=lambda: 0)
    manager = StalenessManager(version, 2, 2, 1)

    def task_factory(task):
        async def complete():
            manager.on_rollout_accepted()
            return _inline_payload(0, rows=4)

        return complete

    dispatcher = BatchTaskDispatcher(16, task_factory, manager)
    dispatcher.initialize(logger=MagicMock())
    source = InlineBatchSource(
        tmp_path / "inline", batch_size=2, group_size=4, max_staleness=1
    )
    finished_batch = threading.Event()

    def inputs():
        for i in range(100):
            yield SimpleNamespace(task_id=i)

    generator = inputs()

    def prepare(**kwargs):
        assert kwargs["group_size"] == 4
        rows = dispatcher.active_submit_and_wait(generator, batch_size=2)
        finished_batch.set()
        return rows

    controller = SimpleNamespace(
        prepare_batch=prepare,
        staleness_manager=manager,
        workflow_executor=SimpleNamespace(dispatcher=dispatcher),
    )
    consumer = source.queue.consumer()
    try:
        source.start(controller, total_batches=2, group_size=4)
        assert finished_batch.wait(5)
        source._thread.join(timeout=5)
        assert not source._thread.is_alive()
        assert source.seal_batch(0, consumer=consumer) == "inline-0"
        assert source.queue.status()["counts"]["pending"] == 2
        assert manager.get_stats().accepted == 4
        assert (
            manager.get_capacity() == 0
        )  # Original serving-version bound still applies.
        first = consumer.claim()
        assert len(first.payload["trajectories"]) == 2
        from areal.infra.rpc.serialization import deserialize_value
        from areal.utils.data import concat_batch

        groups = [deserialize_value(row) for row in first.payload["trajectories"]]
        assert [row["input_ids"].shape[0] for row in groups] == [4, 4]
        assert concat_batch(groups)[1].traj_group_sizes == [4, 4]
        import torch

        from areal.api.cli_args import NormConfig
        from areal.utils.data import Normalization

        groups[0]["rewards"].zero_()
        tensors, metadata = concat_batch(groups)
        normalize = Normalization(
            NormConfig(mean_level="group", std_level="group", group_size=4)
        )
        normalized = normalize(
            tensors["rewards"], group_sizes=metadata.traj_group_sizes
        )
        # Constant-reward groups both have zero advantage. Batch normalization
        # would incorrectly reward one task at the other's expense.
        torch.testing.assert_close(normalized, torch.zeros(8), rtol=0, atol=0)
        status = source.status()
        assert status["count_unit"] == "task_groups"
        assert status["group_size"] == 4
        assert status["counts"]["ready"] == status["counts"]["reserved"] == 2
        consumer.mark_applied(first)
        source.confirm_publication(1, consumer=consumer)
        assert source.seal_batch(1, consumer=consumer) == "inline-1"
    finally:
        source.close()
        dispatcher.destroy()
        consumer.close()


@pytest.mark.parametrize("rows", [3, 5])
def test_inline_rejects_partial_or_branched_group_before_enqueue(tmp_path, rows):
    from dataclasses import dataclass

    @dataclass
    class Stats:
        accepted: int = 2

    source = InlineBatchSource(
        tmp_path / "inline", batch_size=2, group_size=4, max_staleness=1
    )
    controller = SimpleNamespace(
        prepare_batch=lambda **kwargs: [
            _inline_payload(rows=4),
            _inline_payload(rows=rows),
        ],
        staleness_manager=SimpleNamespace(get_stats=Stats),
        workflow_executor=SimpleNamespace(dispatcher=MagicMock()),
    )
    consumer = source.queue.consumer()
    try:
        source.start(controller, total_batches=1, group_size=4)
        source._thread.join(timeout=5)
        with pytest.raises(RuntimeError, match="producer failed"):
            source.seal_batch(0, consumer=consumer)
        assert source.queue.status()["counts"]["pending"] == 0
    finally:
        source.close()
        consumer.close()


def test_native_batch_wait_exits_on_normal_shutdown():
    manager = StalenessManager(SimpleNamespace(get_version=lambda: 0), 1, 1, 0)
    dispatcher = BatchTaskDispatcher(16, lambda task: None, manager)
    started, stopped = threading.Event(), threading.Event()

    def wait():
        started.set()
        try:
            dispatcher.wait_results(1, timeout=30)
        except RuntimeError as error:
            assert "shutting down" in str(error)
            stopped.set()

    thread = threading.Thread(target=wait)
    thread.start()
    assert started.wait(1)
    dispatcher.request_shutdown()
    assert stopped.wait(2)
    thread.join(timeout=2)
    assert not thread.is_alive()
