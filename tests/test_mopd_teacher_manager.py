# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.api import SaveLoadMeta
from areal.api.cli_args import (
    MOPDConfig,
    MOPDTeacherManagerConfig,
    MOPDTeacherSpec,
)
from areal.trainer.mopd.targets import MOPD_CONTRIBUTIONS_KEY, aggregate_mopd_targets
from areal.trainer.mopd.teacher_manager import (
    DiskCheckpointProvider,
    DrainReceipt,
    PersistentTeacherManager,
    TeacherManagerState,
)
from areal.trainer.rl_trainer import PPOTrainer


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
) -> MOPDConfig:
    return MOPDConfig(
        teachers={
            teacher_id: MOPDTeacherSpec(path=str(path))
            for teacher_id, path in checkpoints.items()
        },
        routes={"route": {teacher_id: 1.0 for teacher_id in checkpoints}},
        manager=MOPDTeacherManagerConfig(type="disk"),
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
    manager.release(DrainReceipt(complete=True))
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
    manager.release(DrainReceipt(complete=True))
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

    manager.release(DrainReceipt(complete=True))
    manager.release(DrainReceipt(complete=True))

    assert controller.events == ["offload"]
    assert manager.state is TeacherManagerState.OFFLOADED
    manager.close()


def test_persistent_manager_rejects_release_before_drain(tmp_path):
    """An incomplete receipt leaves the resident teacher untouched."""
    t0 = _write_checkpoint(tmp_path, "t0", b"first")
    controller = _PersistentController()
    manager = PersistentTeacherManager(_config({"t0": t0}), lambda _: controller)
    manager.load("t0")

    with pytest.raises(RuntimeError, match="before actor RTensor drain"):
        manager.release(DrainReceipt(complete=False))

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
        manager.release(DrainReceipt(complete=True))
        controller.events.clear()
    controller.fail_on = failure

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        if failure == "offload":
            manager.release(DrainReceipt(complete=True))
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
        manager.release(DrainReceipt(complete=True))
    elif close_state == "broken":
        controller.fail_on = "offload"
        with pytest.raises(RuntimeError, match="offload failed"):
            manager.release(DrainReceipt(complete=True))

    manager.close()
    manager.close()

    assert controller.destroy_calls == 1
    assert manager.state is TeacherManagerState.CLOSED


def test_disk_provider_requires_existing_local_snapshot(tmp_path):
    """Disk mode rejects missing paths instead of attempting a network fetch."""
    provider = DiskCheckpointProvider(_config({"missing": tmp_path / "missing"}))

    with pytest.raises(FileNotFoundError, match="not a local directory"):
        provider.resolve("missing")


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
        return {
            "complete": True,
            "source_shards_cleared": sum(len(target) for target in targets),
            "actor_fetch_buffers_cleared": 2,
        }


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

    def release(self, receipt: DrainReceipt) -> None:
        assert receipt.complete
        self.events.append("release")

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _PhaseActor:
    def __init__(self, events: list[str]):
        self.events = events

    def aggregate_mopd_targets(self, batch, **coefficients):
        self.events.append("aggregate")
        return aggregate_mopd_targets(batch, **coefficients)

    def assert_mopd_runtime_topology(self) -> None:
        self.events.append("topology:actor")

    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:actor:{target_sizes}")
        return {
            "complete": True,
            "source_shards_cleared": sum(len(target) for target in targets),
            "actor_fetch_buffers_cleared": 2,
        }


class _PhaseCritic:
    def __init__(self, events: list[str]):
        self.events = events

    def strict_clear_batches(self, *targets):
        target_sizes = ",".join(str(len(target)) for target in targets)
        self.events.append(f"clear:critic:{target_sizes}")
        return {
            "complete": True,
            "source_shards_cleared": sum(len(target) for target in targets),
            "actor_fetch_buffers_cleared": 2,
        }


class _FailingPhaseActor(_PhaseActor):
    def aggregate_mopd_targets(self, batch, **coefficients):
        del batch, coefficients
        self.events.append("aggregate")
        raise RuntimeError("aggregation failed")


def test_trainer_mopd_phase_routes_reuses_drains_then_releases():
    """Teacher scoring isolates prior contributions before aggregation and release."""
    events: list[str] = []
    mopd = MOPDConfig(
        teachers={
            "t0": MOPDTeacherSpec(path="/unused/t0"),
            "t1": MOPDTeacherSpec(path="/unused/t1"),
        },
        routes={"r0": {"t0": 2.0}, "r1": {"t0": 0.5, "t1": 1.5}},
    )
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(mopd=mopd)
    trainer.mopd_teacher_manager = _PhaseManager(events)
    trainer.actor = _PhaseActor(events)
    trainer.critic = _PhaseCritic(events)
    batch = [{"mopd_route": "r0"}, {"mopd_route": "r1"}]

    result = trainer._run_mopd_teacher_phase(batch)

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
        routes={"r0": {"t0": 1.0}},
    )
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(mopd=mopd)
    trainer.mopd_teacher_manager = _PhaseManager(events)
    trainer.actor = _FailingPhaseActor(events)
    trainer.critic = _PhaseCritic(events)

    with pytest.raises(RuntimeError, match="aggregation failed"):
        trainer._run_mopd_teacher_phase([{"mopd_route": "r0"}])

    assert events == [
        "topology:actor",
        "prefetch:t0",
        "load:t0",
        "topology:t0",
        "compute:t0:1",
        "aggregate",
        "clear:critic:1",
        "clear:t0:1,1",
        "clear:actor:1,1",
        "release",
    ]
