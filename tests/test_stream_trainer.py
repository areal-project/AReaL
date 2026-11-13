# SPDX-License-Identifier: Apache-2.0

"""Test streamed execution ordering with a collector gated on real backward."""

import queue
import threading
from types import SimpleNamespace

import pytest

from areal.api.cli_args import StreamTrainerConfig
from areal.trainer.stream_trainer import StreamTrainer


class _Actor:
    def __init__(self, collection_returned, *, fail_backward=False):
        self.workers_is_dp_head = [True, True]
        self.collection_returned = collection_returned
        self.first_backward = threading.Event()
        self.fail_backward = fail_backward
        self.accumulated = []
        self.commits = []
        self.aborts = []
        self.lock = threading.Lock()

    def stream_topology(self):
        # Physical placement deliberately reverses the rollout rank order.
        return [
            {
                "rank": 0,
                "dp_rank": 0,
                "replica_ranks": [0],
                "host": "node",
                "gpu_uuid": "GPU-slow",
            },
            {
                "rank": 1,
                "dp_rank": 1,
                "replica_ranks": [1],
                "host": "node",
                "gpu_uuid": "GPU-fast",
            },
        ]

    def stream_accumulate(self, update_id, chunk_id, data, *, worker_ranks):
        with self.lock:
            self.accumulated.append((update_id, chunk_id, data, worker_ranks))
        if chunk_id == 10:
            assert not self.collection_returned.is_set(), (
                "Fast-group backward must execute before slow-group collection ends"
            )
            self.first_backward.set()
            if self.fail_backward:
                raise RuntimeError("injected backward failure")

    def finish_stream_update(self, update_id):
        assert self.collection_returned.is_set()
        self.commits.append(update_id)

    def abort_stream_update(self, update_id):
        self.aborts.append(update_id)


class _Rollout:
    def __init__(self):
        self.workers = [SimpleNamespace(ip="node"), SimpleNamespace(ip="node")]
        self.events = queue.Queue()
        self.closed = set()
        self.finished_workers = set()
        self.collecting = False
        self.pauses = 0
        self.restores = 0
        self.final_drains = 0
        self.resumes = 0

    def stream_topology(self):
        return [
            {"worker_rank": 0, "host": "node", "gpu_uuids": ["GPU-fast"]},
            {"worker_rank": 1, "host": "node", "gpu_uuids": ["GPU-slow"]},
        ]

    def get_version(self):
        return 7

    def begin_stream_collection(self):
        assert not self.collecting
        self.collecting = True

    def get_stream_completed(self, timeout=0.0):
        try:
            return [self.events.get(timeout=timeout)]
        except queue.Empty:
            return []

    def try_drain_stream_replica(self, worker_rank, keep_one=True):
        return self.try_drain_stream_replicas([worker_rank], keep_one=keep_one)

    def try_drain_stream_replicas(self, worker_ranks, keep_one=True):
        assert self.collecting
        ranks = set(worker_ranks)
        if not ranks <= self.finished_workers:
            return False
        if keep_one and len(self.closed | ranks) >= len(self.workers):
            return False
        self.closed.update(ranks)
        return True

    def finish_stream_draining(self):
        assert self.finished_workers == set(range(len(self.workers)))
        self.final_drains += 1
        self.closed.update(self.finished_workers)
        return sorted(self.closed)

    def pause(self):
        self.pauses += 1

    def resume(self):
        assert self.collecting, "Observe reservations before admitting prefetched work"
        self.resumes += 1

    def end_stream_collection(self):
        self.collecting = False

    def restore_stream_replicas(self):
        self.restores += 1
        self.closed.clear()


def _experiment(monkeypatch, *, fail_backward=False, wrong_membership=False):
    """The slow collector cannot finish unless early backward actually runs."""
    monkeypatch.setattr(
        "areal.trainer.stream_trainer.stats_tracker.scalar", lambda **_: None
    )
    collection_returned = threading.Event()
    actor = _Actor(collection_returned, fail_backward=fail_backward)
    rollout = _Rollout()
    fast, slow = {"sample": "fast"}, {"sample": "slow"}
    trainer = StreamTrainer(
        actor, rollout, StreamTrainerConfig(enabled=True, poll_interval=0.001)
    )

    def prepare_batch():
        rollout.finished_workers.add(0)
        rollout.events.put({"task_id": 10, "worker_rank": 0, "trajectory": fast})
        if not actor.first_backward.wait(timeout=5):
            raise AssertionError(
                "Collector received no early backward within 5 seconds"
            )
        rollout.finished_workers.add(1)
        rollout.events.put({"task_id": 20, "worker_rank": 1, "trajectory": slow})
        collection_returned.set()
        return [fast, {"sample": "impostor"} if wrong_membership else slow]

    return trainer, actor, rollout, prepare_batch, fast, slow


def test_stream_trainer_fast_backward_overlaps_collection_without_early_commit(
    monkeypatch,
):
    """An early gradient runs while rollout is live; optimizer waits for commit."""
    trainer, actor, rollout, prepare, fast, slow = _experiment(monkeypatch)

    batch = trainer.collect_and_accumulate(prepare, global_step=3)

    assert batch[0] is fast and batch[1] is slow
    assert actor.first_backward.is_set()
    assert sorted(call[1] for call in actor.accumulated) == [10, 20]
    assert actor.commits == []
    assert actor.aborts == []
    assert rollout.final_drains == 1
    assert not rollout.collecting
    trainer.commit()
    assert actor.commits == ["step-3-v7"]
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()
    assert len(actor.commits) == 1


def test_stream_trainer_gpu_uuid_mapping_overrides_rank_order(monkeypatch):
    """Rollout worker zero must switch actor rank one on its physical GPU."""
    trainer, actor, _, prepare, fast, _ = _experiment(monkeypatch)

    trainer.collect_and_accumulate(prepare, global_step=3)

    (fast_call,) = [call for call in actor.accumulated if call[1] == 10]
    assert fast_call[2][0] is fast
    assert fast_call[3] == [1]


def test_stream_trainer_wrong_membership_aborts_and_forbids_optimizer(monkeypatch):
    """Even equal-sized batches cannot replace a previously observed group."""
    trainer, actor, rollout, prepare, _, _ = _experiment(
        monkeypatch, wrong_membership=True
    )

    with pytest.raises(RuntimeError, match="exactly match"):
        trainer.collect_and_accumulate(prepare, global_step=3)

    assert actor.aborts == ["step-3-v7"]
    assert actor.commits == []
    assert rollout.pauses >= 1
    assert not rollout.collecting
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()


def test_stream_trainer_failed_backward_aborts_and_forbids_optimizer(monkeypatch):
    """A partially modified gradient buffer can never enter ordinary commit."""
    trainer, actor, rollout, prepare, _, _ = _experiment(
        monkeypatch, fail_backward=True
    )

    with pytest.raises(RuntimeError, match="injected backward failure"):
        trainer.collect_and_accumulate(prepare, global_step=3)

    assert actor.aborts == ["step-3-v7"]
    assert actor.commits == []
    assert rollout.pauses >= 1
    assert not rollout.collecting
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()


def test_stream_trainer_release_requires_commit_before_reopening_rollout(monkeypatch):
    """Pending gradients prevent inference admission from reopening."""
    trainer, actor, rollout, prepare, _, _ = _experiment(monkeypatch)
    trainer.collect_and_accumulate(prepare, global_step=3)
    with pytest.raises(RuntimeError, match="before a streamed update commits"):
        trainer.release_rollout()
    assert rollout.restores == 0

    trainer.commit()
    trainer.release_rollout()

    assert len(actor.commits) == 1
    assert rollout.restores == 1
    assert not rollout.closed


def test_stream_topology_nonhead_rpc_none_preserves_complete_tp_mapping():
    """The training RPC guard returns None for non-DP-head queries."""
    actor = SimpleNamespace(
        workers_is_dp_head=[True, None, True, None],
        stream_topology=lambda: [
            dict(
                rank=rank,
                dp_rank=rank // 2,
                replica_ranks=[rank // 2 * 2, rank // 2 * 2 + 1],
                host="node",
                gpu_uuid=f"GPU-{rank}",
            )
            for rank in range(4)
        ],
    )
    rollout = SimpleNamespace(
        workers=[SimpleNamespace(ip="node"), SimpleNamespace(ip="node")],
        stream_topology=lambda: [
            dict(worker_rank=0, gpu_uuids=["GPU-2", "GPU-3"]),
            dict(worker_rank=1, gpu_uuids=["GPU-0", "GPU-1"]),
        ],
    )
    trainer = StreamTrainer(actor, rollout, StreamTrainerConfig(enabled=True))

    trainer._discover_replicas()

    assert trainer._replicas[0].actor_ranks == [0, 1]
    assert trainer._replicas[0].rollout_ranks == [1]
    assert trainer._replicas[1].actor_ranks == [2, 3]
    assert trainer._replicas[1].rollout_ranks == [0]
    assert trainer._server_replicas == {0: 1, 1: 0}


def _paired_topology():
    actor = SimpleNamespace(
        workers_is_dp_head=[True, None, None, None, True, None, None, None],
        stream_topology=lambda: [
            dict(
                rank=rank,
                dp_rank=rank // 4,
                replica_ranks=list(range(rank // 4 * 4, rank // 4 * 4 + 4)),
                host="node",
                gpu_uuid=f"GPU-{rank}",
            )
            for rank in range(8)
        ],
    )
    servers = [
        dict(worker_rank=rank, gpu_uuids=[f"GPU-{gpu}" for gpu in gpus])
        for rank, gpus in enumerate([(4, 5), (0, 1), (6, 7), (2, 3)])
    ]
    return actor, servers


def test_stream_topology_tp4_actor_maps_pairs_by_physical_gpu_union():
    """The two TP2 servers for a TP4 actor need not have consecutive ranks."""
    actor, servers = _paired_topology()
    rollout = SimpleNamespace(
        workers=[SimpleNamespace(ip="node") for _ in range(4)],
        stream_topology=lambda: servers,
    )
    trainer = StreamTrainer(actor, rollout, StreamTrainerConfig(enabled=True))

    trainer._discover_replicas()

    assert trainer._replicas[0].actor_ranks == [0, 1, 2, 3]
    assert trainer._replicas[0].rollout_ranks == [1, 3]
    assert trainer._replicas[1].actor_ranks == [4, 5, 6, 7]
    assert trainer._replicas[1].rollout_ranks == [0, 2]


@pytest.mark.parametrize("invalid", ["cross_actor", "overlap", "missing"])
def test_stream_topology_invalid_inference_partition_rejected(invalid):
    """Every server must lie in one actor and cover its GPUs exactly once."""
    actor, servers = _paired_topology()
    if invalid == "cross_actor":
        servers[0]["gpu_uuids"] = ["GPU-0", "GPU-4"]
    elif invalid == "overlap":
        servers[0]["gpu_uuids"] = servers[1]["gpu_uuids"]
    else:
        servers[0]["gpu_uuids"] = ["GPU-4"]
    rollout = SimpleNamespace(
        workers=[SimpleNamespace(ip="node") for _ in range(4)],
        stream_topology=lambda: servers,
    )
    trainer = StreamTrainer(actor, rollout, StreamTrainerConfig(enabled=True))

    with pytest.raises(RuntimeError, match="actor TP replica|overlap|partition"):
        trainer._discover_replicas()


def test_stream_trainer_paired_servers_wait_for_whole_actor_then_overlap(monkeypatch):
    """Two unpaired free servers cannot start a TP4 actor on live GPU memory."""
    monkeypatch.setattr(
        "areal.trainer.stream_trainer.stats_tracker.scalar", lambda **_: None
    )
    topology, servers = _paired_topology()
    collection_returned = threading.Event()
    partial_seen = threading.Event()
    actor = _Actor(collection_returned)
    actor.workers_is_dp_head = topology.workers_is_dp_head
    actor.stream_topology = topology.stream_topology
    rollout = _Rollout()
    rollout.workers = [SimpleNamespace(ip="node") for _ in range(4)]
    rollout.stream_topology = lambda: servers
    original_get = rollout.get_stream_completed
    consumed = 0

    def get_completed(timeout=0.0):
        nonlocal consumed
        if consumed == 2:
            partial_seen.set()
        events = original_get(timeout)
        consumed += len(events)
        return events

    rollout.get_stream_completed = get_completed
    groups = [{"group": index} for index in range(4)]

    def publish(worker, chunk, group):
        rollout.finished_workers.add(worker)
        rollout.events.put(
            {"task_id": chunk, "worker_rank": worker, "trajectory": group}
        )

    def prepare():
        publish(1, 10, groups[0])
        publish(0, 20, groups[1])
        assert partial_seen.wait(timeout=5)
        assert not rollout.closed
        assert actor.accumulated == []
        publish(3, 30, groups[2])
        assert actor.first_backward.wait(timeout=5)
        assert rollout.closed == {1, 3}
        publish(2, 40, groups[3])
        collection_returned.set()
        return groups

    trainer = StreamTrainer(
        actor, rollout, StreamTrainerConfig(enabled=True, poll_interval=0.001)
    )

    assert trainer.collect_and_accumulate(prepare, global_step=0) == groups
    trainer.commit()

    assert sorted(call[1] for call in actor.accumulated) == [10, 20, 30, 40]
    assert all(call[3] in ([0, 1, 2, 3], [4, 5, 6, 7]) for call in actor.accumulated)
    assert actor.commits == ["step-0-v7"]


def test_stream_trainer_consecutive_steps_observe_before_resuming_prefetch(monkeypatch):
    trainer, actor, rollout, prepare, _, _ = _experiment(monkeypatch)

    for step in range(2):
        actor.collection_returned.clear()
        actor.first_backward.clear()
        rollout.finished_workers.clear()
        trainer.collect_and_accumulate(prepare, global_step=step)
        trainer.commit()
        trainer.release_rollout()

    assert rollout.resumes == 2
    assert actor.commits == ["step-0-v7", "step-1-v7"]
    assert rollout.restores == 2
