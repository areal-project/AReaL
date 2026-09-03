# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from areal.api import SaveLoadMeta
from areal.api.cli_args import (
    MOPDConfig,
    MOPDTeacherManagerConfig,
    MOPDTeacherSpec,
)
from areal.infra.rpc.rtensor import RTensorDrainReceipt
from areal.trainer.mopd.targets import MOPD_CONTRIBUTIONS_KEY, aggregate_mopd_targets
from areal.trainer.mopd.teacher_manager import (
    DiskCheckpointProvider,
    LocalMemoryCheckpointProvider,
    PersistentTeacherManager,
    TeacherManagerState,
)
from areal.trainer.mopd.teacher_phase import MOPDTeacherPhase


def _receipt(role: str = "actor") -> RTensorDrainReceipt:
    return RTensorDrainReceipt(
        consumer_role=role,
        shard_count=1,
        source_node_count=1,
        consumer_dp_head_count=1,
    )


def _write_checkpoint(root: Path, teacher_id: str, payload: bytes) -> Path:
    checkpoint = root / teacher_id
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"teacher": teacher_id}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(payload)
    return checkpoint


def _config(
    checkpoints: dict[str, Path],
    *,
    manager_type: str = "disk",
    staging_root: Path | None = None,
) -> MOPDConfig:
    return MOPDConfig(
        teachers={
            teacher_id: MOPDTeacherSpec(path=str(path))
            for teacher_id, path in checkpoints.items()
        },
        teacher_groups={"group": {teacher_id: 1.0 for teacher_id in checkpoints}},
        manager=MOPDTeacherManagerConfig(
            type=manager_type,
            staging_root=str(staging_root or "/unused"),
        ),
    )


@dataclass
class _PersistentController:
    events: list[str] = field(default_factory=list)
    fail_on: str | None = None
    destroy_calls: int = 0
    destroy_failures: int = 0

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed")

    def onload(self) -> None:
        self._event("onload")

    def offload(self) -> None:
        self._event("offload")

    def load(self, meta: SaveLoadMeta) -> None:
        self._event(f"load:{Path(meta.path).name}")
        assert meta.with_optim is False

    def destroy(self) -> None:
        self.destroy_calls += 1
        self.events.append("destroy")
        if self.destroy_failures:
            self.destroy_failures -= 1
            raise RuntimeError("destroy failed")


def test_persistent_manager_reuses_controller_across_phases_and_checkpoints(tmp_path):
    """Phase boundaries offload/onload one companion instead of respawning it."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    t1 = _write_checkpoint(tmp_path, "t1", b"second")
    controller = _PersistentController()
    factory_paths: list[str] = []

    def factory(path: str) -> _PersistentController:
        factory_paths.append(path)
        return controller

    manager = PersistentTeacherManager(_config({"t0": t0, "t1": t1}), factory)

    first = manager.load("t0")
    manager.release(_receipt())
    second = manager.load("t0")
    third = manager.load("t1")

    assert first is second is third is controller
    assert factory_paths == [str(t0)]
    assert controller.events == ["offload", "onload", "load:t1"]
    assert controller.destroy_calls == 0
    assert manager.state is TeacherManagerState.RESIDENT
    manager.close()
    assert controller.destroy_calls == 1


def test_persistent_manager_onloads_before_cross_phase_checkpoint_switch(tmp_path):
    """An offloaded companion becomes resident before loading another teacher."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    t1 = _write_checkpoint(tmp_path, "t1", b"second")
    controller = _PersistentController()
    manager = PersistentTeacherManager(
        _config({"t0": t0, "t1": t1}), lambda _: controller
    )
    manager.load("t0")
    manager.release(_receipt())
    controller.events.clear()

    manager.load("t1")

    assert controller.events == ["onload", "load:t1"]
    assert manager.state is TeacherManagerState.RESIDENT
    manager.close()


def test_persistent_manager_repeated_release_does_not_offload_twice(tmp_path):
    """An already-offloaded companion treats a duplicate complete receipt as a noop."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    controller = _PersistentController()
    manager = PersistentTeacherManager(_config({"t0": t0}), lambda _: controller)
    manager.load("t0")

    manager.release(_receipt())
    manager.release(_receipt())

    assert controller.events == ["offload"]
    assert manager.state is TeacherManagerState.OFFLOADED
    manager.close()


def test_persistent_manager_does_not_restage_unchanged_local_checkpoint(tmp_path):
    """An offloaded unchanged teacher resumes without copying its snapshot again."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    t0 = _write_checkpoint(source_root, "t0", b"first")
    controller = _PersistentController()
    manager = PersistentTeacherManager(
        _config(
            {"t0": t0},
            manager_type="local_memory",
            staging_root=tmp_path / "staging",
        ),
        lambda _: controller,
    )
    manager.pre_fetch("t0")
    manager.load("t0")
    manager.release(_receipt())

    manager.pre_fetch("t0")
    manager.load("t0")

    assert controller.events == ["offload", "onload"]
    manager.close()


def test_persistent_manager_rejects_non_actor_receipt(tmp_path):
    """A non-actor receipt leaves the resident teacher untouched."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    controller = _PersistentController()
    manager = PersistentTeacherManager(_config({"t0": t0}), lambda _: controller)
    manager.load("t0")

    with pytest.raises(RuntimeError, match="without an actor RTensor drain receipt"):
        manager.release(_receipt("teacher"))

    assert controller.events == []
    assert manager.state is TeacherManagerState.RESIDENT
    manager.close()


@pytest.mark.parametrize(
    ("failure", "prepare"),
    [
        ("onload", "offload"),
        ("load:t1", "resident"),
        ("offload", "resident"),
    ],
)
def test_persistent_manager_failure_destroys_companion(tmp_path, failure, prepare):
    """Load, onload, and offload failures poison the whole group."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    t1 = _write_checkpoint(tmp_path, "t1", b"second")
    controller = _PersistentController()
    manager = PersistentTeacherManager(
        _config({"t0": t0, "t1": t1}), lambda _: controller
    )
    manager.load("t0")
    if prepare == "offload":
        manager.release(_receipt())
        controller.events.clear()
    controller.fail_on = failure

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        if failure == "offload":
            manager.release(_receipt())
        elif failure == "load:t1":
            manager.load("t1")
        else:
            manager.load("t0")

    assert controller.destroy_calls == 1
    assert controller.events[-1] == "destroy"
    assert manager.controller is None
    assert manager.state is TeacherManagerState.BROKEN
    manager.close()
    assert controller.destroy_calls == 1


@pytest.mark.parametrize("close_state", ["resident", "offloaded", "broken"])
def test_persistent_manager_close_is_idempotent_in_every_live_state(
    tmp_path, close_state
):
    """Closing any persistent state tears down the companion at most once."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    controller = _PersistentController()
    manager = PersistentTeacherManager(_config({"t0": t0}), lambda _: controller)
    manager.load("t0")
    if close_state == "offloaded":
        manager.release(_receipt())
    elif close_state == "broken":
        controller.fail_on = "offload"
        with pytest.raises(RuntimeError, match="offload failed"):
            manager.release(_receipt())

    manager.close()
    manager.close()

    assert controller.destroy_calls == 1
    assert manager.state is TeacherManagerState.CLOSED


def test_disk_provider_requires_existing_local_snapshot(tmp_path):
    """Disk mode rejects missing paths instead of attempting a network fetch."""
    provider = DiskCheckpointProvider(_config({"missing": tmp_path / "missing"}))

    with pytest.raises(FileNotFoundError, match="not a local directory"):
        provider.resolve("missing")


def test_local_memory_provider_uses_atomic_single_ready_checkpoint(tmp_path):
    """Staging publishes one ready snapshot and removes it after consumption."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    t0 = _write_checkpoint(source_root, "t0", b"first")
    t1 = _write_checkpoint(source_root, "t1", b"second")
    staging_root = tmp_path / "staging"
    provider = LocalMemoryCheckpointProvider(
        _config(
            {"t0": t0, "t1": t1},
            manager_type="local_memory",
            staging_root=staging_root,
        )
    )

    provider.pre_fetch("t0")
    ready = provider.resolve("t0")

    assert ready.name == "t0.ready"
    assert (ready / "model.safetensors").read_bytes() == b"first"
    assert not list(staging_root.rglob("*.tmp.*"))
    with pytest.raises(RuntimeError, match="already holds ready checkpoint"):
        provider.pre_fetch("t1")

    provider.consumed("t0")
    assert not ready.exists()
    provider.pre_fetch("t1")
    second = provider.resolve("t1")
    assert (second / "model.safetensors").read_bytes() == b"second"
    provider.close()
    provider.close()
    assert not list(staging_root.iterdir())


def test_local_memory_provider_rejects_insufficient_capacity(tmp_path, monkeypatch):
    """Capacity is checked before checkpoint bytes enter the staging root."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    t0 = _write_checkpoint(source_root, "t0", b"payload")
    staging_root = tmp_path / "staging"
    provider = LocalMemoryCheckpointProvider(
        _config(
            {"t0": t0},
            manager_type="local_memory",
            staging_root=staging_root,
        )
    )
    monkeypatch.setattr(
        "areal.trainer.mopd.teacher_manager.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": 0})(),
    )

    with pytest.raises(OSError, match="Insufficient staging space"):
        provider.pre_fetch("t0")

    provider.close()
    assert not list(staging_root.iterdir())


def test_local_memory_provider_sweeps_dead_run(tmp_path):
    """Construction removes run directories whose owning process is gone."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    t0 = _write_checkpoint(source_root, "t0", b"payload")
    staging_root = tmp_path / "staging"
    stale = staging_root / ".run-stale"
    stale.mkdir(parents=True)
    (stale / "owner.json").write_text('{"pid": 999999999}', encoding="utf-8")
    (stale / "orphan.tmp.data").write_bytes(b"orphan")

    provider = LocalMemoryCheckpointProvider(
        _config(
            {"t0": t0},
            manager_type="local_memory",
            staging_root=staging_root,
        )
    )

    assert not stale.exists()
    provider.close()


class _PhaseController:
    def __init__(self, teacher_id: str, events: list[str]):
        self.teacher_id = teacher_id
        self.events = events

    def compute_logp_padded(self, subset):
        self.events.append(f"compute:{self.teacher_id}:{len(subset)}")
        assert all(MOPD_CONTRIBUTIONS_KEY not in trajectory for trajectory in subset)
        value = 1.0 if self.teacher_id == "t0" else 3.0
        real = [torch.full((2,), value) for _ in subset]
        dummy = [torch.full((1,), -1.0)] if self.teacher_id == "t1" else []
        return real, dummy

    def assert_mopd_runtime_topology(self) -> None:
        self.events.append(f"topology:{self.teacher_id}")

    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:{self.teacher_id}:{target_sizes}")
        return _receipt("mopd-teacher")


class _PhaseManager:
    def __init__(self, events: list[str]):
        self.events = events
        self.closed = False
        self.state = TeacherManagerState.RESIDENT

    def pre_fetch(self, teacher_id: str) -> None:
        self.events.append(f"prefetch:{teacher_id}")

    def load(self, teacher_id: str) -> _PhaseController:
        self.events.append(f"load:{teacher_id}")
        return _PhaseController(teacher_id, self.events)

    def release(self, receipt: RTensorDrainReceipt) -> None:
        assert receipt.consumer_role == "actor"
        self.events.append("release")

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _PhaseActor:
    def __init__(self, events: list[str]):
        self.events = events

    def aggregate_mopd_targets(self, batch):
        self.events.append("aggregate")
        return aggregate_mopd_targets(batch)

    def assert_mopd_runtime_topology(self) -> None:
        self.events.append("topology:actor")

    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:actor:{target_sizes}")
        return _receipt("actor")


class _PhaseCritic:
    def __init__(self, events: list[str]):
        self.events = events

    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:critic:{target_sizes}")
        return _receipt("critic")


class _PhaseRef(_PhaseCritic):
    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:ref:{target_sizes}")
        return _receipt("ref")


class _FailingPhaseActor(_PhaseActor):
    def aggregate_mopd_targets(self, batch):
        del batch
        self.events.append("aggregate")
        raise RuntimeError("aggregation failed")


class _FailingPhaseRef(_PhaseRef):
    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:ref:{target_sizes}")
        raise RuntimeError("ref drain failed")


def test_trainer_mopd_phase_routes_reuses_drains_then_releases():
    """Teacher scoring isolates prior contributions before aggregation and release."""
    events: list[str] = []
    mopd = MOPDConfig(
        teachers={
            "t0": MOPDTeacherSpec(path="/unused/t0"),
            "t1": MOPDTeacherSpec(path="/unused/t1"),
        },
        teacher_groups={"r0": {"t0": 2.0}, "r1": {"t0": 0.5, "t1": 1.5}},
    )
    phase = MOPDTeacherPhase(
        config=mopd,
        manager=_PhaseManager(events),
        actor=_PhaseActor(events),
        critic=_PhaseCritic(events),
        ref=_PhaseRef(events),
    )
    batch = [{"mopd_route": "r0"}, {"mopd_route": "r1"}]

    result = phase.materialize(batch)

    assert events == [
        "topology:actor",
        "prefetch:t0",
        "load:t0",
        "topology:t0",
        "prefetch:t1",
        "compute:t0:2",
        "load:t1",
        "topology:t1",
        "compute:t1:1",
        "aggregate",
        "clear:critic:2",
        "clear:ref:2",
        "clear:t0:2,4",
        "clear:t1:2,4",
        "clear:actor:2,4",
        "release",
    ]
    torch.testing.assert_close(
        result[0]["mopd_teacher_logp_sum"],
        torch.full((2,), 2.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result[1]["mopd_teacher_logp_sum"],
        torch.full((2,), 5.0),
        rtol=0.0,
        atol=0.0,
    )
    assert all("mopd_route" not in trajectory for trajectory in result)


def test_trainer_mopd_phase_failure_drains_rollout_before_release():
    """Emergency cleanup drains original shards from teacher and actor workers."""
    events: list[str] = []
    mopd = MOPDConfig(
        teachers={"t0": MOPDTeacherSpec(path="/unused/t0")},
        teacher_groups={"r0": {"t0": 1.0}},
    )
    phase = MOPDTeacherPhase(
        config=mopd,
        manager=_PhaseManager(events),
        actor=_FailingPhaseActor(events),
        critic=_PhaseCritic(events),
        ref=_PhaseRef(events),
    )

    with pytest.raises(RuntimeError, match="aggregation failed"):
        phase.materialize([{"mopd_route": "r0"}])

    assert events == [
        "topology:actor",
        "prefetch:t0",
        "load:t0",
        "topology:t0",
        "compute:t0:1",
        "aggregate",
        "clear:critic:1",
        "clear:ref:1",
        "clear:t0:1,1",
        "clear:actor:1,1",
        "release",
    ]


def test_mopd_phase_closes_teacher_when_any_consumer_drain_has_no_ack():
    """A missing consumer ACK forces teardown instead of teacher offload."""
    events: list[str] = []
    mopd = MOPDConfig(
        teachers={"t0": MOPDTeacherSpec(path="/unused/t0")},
        teacher_groups={"r0": {"t0": 1.0}},
    )
    phase = MOPDTeacherPhase(
        config=mopd,
        manager=_PhaseManager(events),
        actor=_PhaseActor(events),
        ref=_FailingPhaseRef(events),
    )

    with pytest.raises(RuntimeError, match="ref drain failed"):
        phase.materialize([{"mopd_route": "r0"}])

    assert "close" in events
    assert "release" not in events
