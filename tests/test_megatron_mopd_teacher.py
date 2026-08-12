# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.api import WeightUpdateMeta, Worker
from areal.engine import MegatronEngine
from areal.infra.controller.train_controller import TrainController
from areal.trainer.rl_trainer import PPOTrainer


def _device_worker_env() -> dict[str, str]:
    """Restore controller-hidden devices for a fresh GPU worker subprocess."""
    env = os.environ.copy()
    hidden_env = env.get("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV")
    if hidden_env:
        if env.get("AREAL_CONTROLLER_ORIG_DEVICES_SET") == "1":
            env[hidden_env] = env.get("AREAL_CONTROLLER_ORIG_DEVICES", "")
        else:
            env.pop(hidden_env, None)
    env["AREAL_ROLE_WORKER"] = "1"
    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not pythonpath else f"{repo_root}:{pythonpath}"
    return env


def _assigned_cuda_devices() -> list[str]:
    original = os.environ.get("AREAL_CONTROLLER_ORIG_DEVICES", "")
    if os.environ.get("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV") == "CUDA_VISIBLE_DEVICES":
        return [device for device in original.split(",") if device]
    return [str(device) for device in range(torch.cuda.device_count())]


def test_mopd_runtime_topology_accepts_configured_pp8(monkeypatch):
    engine = object.__new__(MegatronEngine)
    engine.parallel_strategy = SimpleNamespace(pipeline_parallel_size=8)
    monkeypatch.setattr(
        "areal.engine.megatron_engine.mpu.get_pipeline_model_parallel_world_size",
        lambda: 8,
    )

    engine.assert_mopd_runtime_topology()


def test_mopd_runtime_topology_rejects_config_runtime_mismatch(monkeypatch):
    engine = object.__new__(MegatronEngine)
    engine.parallel_strategy = SimpleNamespace(pipeline_parallel_size=8)
    monkeypatch.setattr(
        "areal.engine.megatron_engine.mpu.get_pipeline_model_parallel_world_size",
        lambda: 4,
    )

    with pytest.raises(RuntimeError, match="configured PP=8, runtime PP=4"):
        engine.assert_mopd_runtime_topology()


def test_teacher_weight_residency_adapter_has_no_awex_publication_state():
    engine = object.__new__(MegatronEngine)
    engine._awex_adapter = None
    engine.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    engine.init_weight_residency_adapter()

    assert engine._awex_adapter is not None
    assert engine._awex_adapter._meta_server_client is None
    assert engine._awex_adapter._initialized is False


def test_awex_actor_worker_does_not_reenter_rollout(monkeypatch):
    events = []

    class _Adapter:
        def execute_colocate_weight_update(self, version):
            events.append(("execute", version))

        def finish_colocate_weight_update(self, training_world_size):
            events.append(("finish", training_world_size))

    class _Rollout:
        def onload(self, tags=None):
            raise AssertionError(f"actor worker re-entered rollout onload: {tags}")

        def continue_generation(self):
            raise AssertionError("actor worker re-entered rollout generation")

    engine = object.__new__(MegatronEngine)
    engine._awex_adapter = _Adapter()
    engine.rollout_engine = _Rollout()
    engine.rollout_coordinator = object()
    engine.process_group_initialized = True
    engine._cpu_group = object()

    monkeypatch.setattr(
        "areal.engine.megatron_engine.dist.barrier",
        lambda group: events.append(("barrier", group)),
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.dist.get_world_size", lambda group: 8
    )

    engine.update_weights(WeightUpdateMeta(type="awex", version=4))

    assert events == [
        ("execute", 4),
        ("barrier", engine.cpu_group),
        ("finish", 8),
        ("barrier", engine.cpu_group),
    ]


def test_awex_controller_discards_unfinished_requests_before_restoring_kv():
    events = []

    class _Actor:
        def update_weights(self, meta):
            events.append(("update", meta.version))

        def set_version(self, version):
            events.append(("actor_version", version))

    class _Rollout:
        def set_version(self, version):
            events.append(("rollout_version", version))

        def abort_all_requests(self):
            events.append(("abort_all", None))

        def onload(self, tags=None):
            events.append(("onload", tags))

        async def continue_generation(self):
            events.append(("continue", None))

    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version="v1", weight_update_mode="awex")
    )
    trainer.actor = _Actor()
    trainer.rollout = _Rollout()
    trainer.critic = None
    trainer.eval_rollout = None

    trainer._update_weights_and_publish_version(
        WeightUpdateMeta(type="awex", version=1), 1
    )
    trainer._update_weights_and_publish_version(
        WeightUpdateMeta(type="awex", version=2), 2
    )

    assert events == [
        ("update", 1),
        ("actor_version", 1),
        ("rollout_version", 1),
        ("abort_all", None),
        ("onload", ["cuda_graph"]),
        ("onload", ["kv_cache"]),
        ("continue", None),
        ("update", 2),
        ("actor_version", 2),
        ("rollout_version", 2),
        ("abort_all", None),
        ("onload", ["cuda_graph"]),
        ("onload", ["kv_cache"]),
        ("continue", None),
    ]


def test_awex_controller_does_not_resume_when_discard_fails():
    events = []

    class _Actor:
        def update_weights(self, meta):
            events.append(("update", meta.version))

        def set_version(self, version):
            events.append(("actor_version", version))

    class _Rollout:
        def set_version(self, version):
            events.append(("rollout_version", version))

        def abort_all_requests(self):
            events.append(("abort_all", None))
            raise RuntimeError("discard failed")

        def onload(self, tags=None):
            raise AssertionError(f"restored KV after discard failure: {tags}")

        def continue_generation(self):
            raise AssertionError("resumed generation after discard failure")

    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version="v1", weight_update_mode="awex")
    )
    trainer.actor = _Actor()
    trainer.rollout = _Rollout()
    trainer.critic = None
    trainer.eval_rollout = None

    with pytest.raises(RuntimeError, match="discard failed"):
        trainer._update_weights_and_publish_version(
            WeightUpdateMeta(type="awex", version=1), 1
        )

    assert events == [
        ("update", 1),
        ("actor_version", 1),
        ("rollout_version", 1),
        ("abort_all", None),
    ]


def test_non_awex_weight_update_does_not_restore_rollout():
    events = []

    class _Actor:
        def update_weights(self, meta):
            events.append(("update", meta.version))

        def set_version(self, version):
            events.append(("actor_version", version))

    class _Rollout:
        def set_version(self, version):
            events.append(("rollout_version", version))

        def onload(self, tags=None):
            raise AssertionError(f"unexpected rollout onload: {tags}")

        def continue_generation(self):
            raise AssertionError("unexpected rollout generation resume")

    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version="v1", weight_update_mode="disk")
    )
    trainer.actor = _Actor()
    trainer.rollout = _Rollout()
    trainer.critic = None
    trainer.eval_rollout = None

    trainer._update_weights_and_publish_version(
        WeightUpdateMeta(type="disk", version=3), 3
    )

    assert events == [
        ("update", 3),
        ("actor_version", 3),
        ("rollout_version", 3),
    ]


@pytest.mark.parametrize("method", ["offload", "onload"])
def test_teacher_lifecycle_collective_waits_all_ranks_without_retry(method):
    """A failed teacher rank cannot trigger a partial collective retry."""
    events = []

    class _Scheduler:
        async def async_call_engine(self, worker_id, method, engine_name, **kwargs):
            events.append(("start", worker_id, method, engine_name, kwargs))
            if worker_id.endswith("/0"):
                raise RuntimeError("rank zero failed")
            await asyncio.sleep(0.01)
            events.append(("complete", worker_id))

    controller = object.__new__(TrainController)
    controller.scheduler = _Scheduler()
    controller._worker_role = "mopd-teacher"
    controller.workers = [
        Worker(id="mopd-teacher/0", ip="127.0.0.1"),
        Worker(id="mopd-teacher/1", ip="127.0.0.1"),
    ]

    with pytest.raises(ExceptionGroup, match=f"collective {method} failed"):
        getattr(controller, method)()

    starts = [event for event in events if event[0] == "start"]
    assert len(starts) == 2
    assert all(event[2] == method for event in starts)
    assert all(event[4]["max_retries"] == 1 for event in starts)
    assert ("complete", "mopd-teacher/1") in events


def test_megatron_teacher_offload_skips_non_ddp_model_chunks(monkeypatch):
    """Scoring teachers can discard DDP grads without assuming every chunk is DDP."""
    events = []

    class _Stats:
        def log(self, message):
            events.append(("stats", message))

    engine = object.__new__(MegatronEngine)
    engine._awex_adapter = None
    engine.mcore_config = SimpleNamespace(disable_grad_buffers_cpu_backup=True)
    engine.model = [object()]
    engine.process_group_initialized = True
    engine._cpu_group = object()
    engine.is_offload = False
    engine.get_device_stats = lambda: _Stats()
    monkeypatch.setattr("areal.engine.megatron_engine.is_tms_enabled", lambda: True)
    monkeypatch.setattr(
        "areal.engine.megatron_engine.current_platform.clear_memory",
        lambda: events.append(("clear", None)),
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.current_platform.synchronize",
        lambda: events.append(("synchronize", None)),
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.torch_memory_saver.pause",
        lambda: events.append(("pause", None)),
    )
    monkeypatch.setattr(
        "areal.engine.megatron_engine.dist.barrier",
        lambda group: events.append(("barrier", group)),
    )

    engine.offload()

    assert engine.is_offload is True
    assert ("pause", None) in events
    assert events.index(("synchronize", None)) < events.index(
        ("barrier", engine.cpu_group)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _assigned_cuda_devices(), reason="requires one CUDA GPU")
@pytest.mark.parametrize("mode", ["fallback", "native"])
def test_teacher_residency_adapter_releases_and_restores_cuda_flat_buffer(mode):
    """Both MCore paths release CUDA storage and preserve teacher weights."""
    runner = Path(__file__).parent / "torchrun" / "run_mopd_teacher_residency.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--mode", mode],
        capture_output=True,
        text=True,
        env=_device_worker_env(),
        cwd=Path(__file__).resolve().parents[1],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CUDA residency worker failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert f"Passed mode={mode}" in result.stdout
