# SPDX-License-Identifier: Apache-2.0

"""Exercise teacher input-cache ownership with real RTensors and local RPC transport."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.controller.train_controller import TrainController
from areal.infra.rpc import rtensor as rt
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActorController
from areal.trainer.rl_trainer import PPOTrainer


def _trajectory() -> dict[str, Any]:
    return {
        "input_ids": torch.ones((1, 4), dtype=torch.int64, device="cpu"),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool, device="cpu"),
        "multi_modal_input": [{"pixel_values": torch.ones((1, 3, 2, 2), device="cpu")}],
    }


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> rt.HttpRTensorBackend:
    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    monkeypatch.setattr(rt, "_storage", {})
    monkeypatch.setattr(rt, "_storage_stats", {})
    monkeypatch.setattr(rt, "_fetch_buffer", {})
    backend = rt.HttpRTensorBackend()
    monkeypatch.setattr(
        backend,
        "fetch",
        Mock(side_effect=lambda shards: [rt.fetch(s.shard_id) for s in shards]),
    )

    async def delete(_addr: str, shard_ids: list[str]) -> dict[str, int]:
        return {"cleared_count": sum(rt.remove(sid) for sid in shard_ids)}

    monkeypatch.setattr(backend, "delete", delete)
    monkeypatch.setattr(rt, "_backend", backend)
    return backend


class _LocalRPCScheduler:
    """Keep serialization/localization real, isolating each worker's fetch cache."""

    def __init__(self, worker_cls: type = RemoteSGLangEngine) -> None:
        self.worker_cls = worker_cls
        self.buffers: dict[str, dict[str, torch.Tensor]] = {
            "teacher/0": {},
            "teacher/1": {},
        }
        self.calls: list[tuple[str, str, tuple, dict]] = []

    async def async_call_engine(
        self, worker_id: str, method: str, engine_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        assert engine_name == worker_id
        self.calls.append((worker_id, method, args, kwargs.copy()))
        kwargs.pop("http_timeout", None)
        rpc_meta = kwargs.pop("rpc_meta", None)
        # These scopes model distinct worker processes without requiring GPUs
        # or sockets. Each coroutine finishes its scope before yielding.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(rt, "_fetch_buffer", self.buffers[worker_id])
            args, kwargs = rt.RTensor.localize(
                deserialize_value(serialize_value((args, kwargs)))
            )
            if method == "compute_logp":
                data = args[0] if args else kwargs["data"]
                result = [
                    torch.zeros_like(item["input_ids"], dtype=torch.float32)
                    for item in data
                ]
                result = rt.RTensor.remotize(result, node_addr=worker_id)
                return deserialize_value(serialize_value(result))
            if method == "clear_batches":
                assert rpc_meta == {"broadcast": False}
                self.worker_cls.clear_batches(None, *args, **kwargs)
                return None
            assert method == "fetch_buffer_stats"
            return rt.fetch_buffer_stats()


def _teacher(scheduler: _LocalRPCScheduler, engine_type: str = "rollout") -> Any:
    if engine_type == "rollout":
        teacher = RolloutController(
            scheduler.worker_cls,
            SimpleNamespace(backend="sglang:d2", request_timeout=10),
            scheduler,
        )
    else:
        teacher = PPOActorController(
            Mock(), SimpleNamespace(backend="fsdp:d2"), scheduler
        )
        teacher.workers_is_dp_head = [True, True]
    teacher._worker_role = "teacher"
    teacher.workers = [SimpleNamespace(id=worker_id) for worker_id in scheduler.buffers]
    return teacher


def _trainer(teacher: Any, engine_type: str = "rollout") -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        total_train_epochs=1,
        total_train_steps=None,
        rollout=SimpleNamespace(agent=None, _version="v1"),
        teacher=SimpleNamespace(
            engine_type=engine_type, rl_loss_weight=1.0, distill_loss_weight=0.01
        ),
        gconfig=SimpleNamespace(
            n_samples=1,
            reward_normalization=False,
            reward_normalization_use_std=True,
            drop_incomplete_group=False,
        ),
        dynamic_bs=False,
        actor=SimpleNamespace(
            _version="v1",
            weight_update_mode="xccl",
            min_usable_group_size=None,
            resolve_min_usable_group_size=lambda _: 1,
            should_compute_prox_logp=lambda: False,
        ),
        memory_profiler=None,
    )
    trainer.recover_info = None
    trainer.train_dataloader = [[{}], [{}], [{}]]
    trainer.saver = Mock()
    trainer.actor = Mock(parallel_strategy=SimpleNamespace(dp_size=1))
    trainer.actor.compute_advantages.return_value = []
    trainer.rollout = Mock()
    trainer.teacher = teacher
    trainer.critic = trainer.ref = trainer.data_controller = None
    trainer.mopd_execution_plan = trainer.mopd_teacher_phase = None
    trainer._should_offload_rollout = trainer._should_offload_actor = False
    trainer._should_offload_teacher = False
    trainer.weight_update_meta = Mock()
    trainer._requires_proxy_workflow = lambda _: False
    trainer._maybe_inject_r3_workflow_kwargs = lambda _, kwargs: kwargs
    trainer._evaluate_before_train = Mock(return_value=False)
    trainer._update_weights_and_publish_version = Mock()
    trainer._save_training_state = Mock()
    trainer._evaluate = Mock()
    trainer._prepare_awex_evaluation = Mock(return_value=None)
    trainer._export_stats_then_restore_awex_rollout = Mock()
    trainer._save_perf_tracer = Mock()
    return trainer


@pytest.mark.parametrize("engine_type", ["rollout", "train"])
def test_teacher_repeated_training_steps_release_owned_cache(
    storage: rt.HttpRTensorBackend, engine_type: str
) -> None:
    """Deleting source shards alone must not leave teacher input caches growing."""
    scheduler = _LocalRPCScheduler()
    teacher = _teacher(scheduler, engine_type)
    trainer = _trainer(teacher, engine_type)
    unrelated = rt.RTensor.remotize(torch.ones(1), node_addr="rollout")
    for buffer in scheduler.buffers.values():
        buffer[unrelated.shard.shard_id] = rt.fetch(unrelated.shard.shard_id)
    baseline = rt.storage_stats()

    def prepare_batch(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert rt.storage_stats() == baseline
        assert all(
            set(buffer) == {unrelated.shard.shard_id}
            for buffer in scheduler.buffers.values()
        )
        return rt.RTensor.remotize([_trajectory(), _trajectory()], node_addr="rollout")

    actor_cleanup = TrainController(Mock(), SimpleNamespace(backend="fsdp:d1"), Mock())
    actor_cleanup._worker_role = "actor"

    def actor_call(method: str, *args: Any, **kwargs: Any) -> Any:
        assert kwargs["rpc_meta"] == {"broadcast": False}
        if method == "clear_batches":
            return rt.clear_fetch_buffer(args[0])
        assert method == "fetch_buffer_stats"
        return rt.fetch_buffer_stats()

    actor_cleanup._custom_function_call = actor_call
    trainer.actor.prepare_batch.side_effect = prepare_batch
    trainer.actor.clear_batches.side_effect = actor_cleanup.clear_batches

    trainer.train(workflow=object())

    assert trainer.actor.ppo_update.call_count == 3
    assert rt.storage_stats() == baseline
    assert all(
        set(buffer) == {unrelated.shard.shard_id}
        for buffer in scheduler.buffers.values()
    )
    assert len([call for call in scheduler.calls if call[1] == "clear_batches"]) == 6


@pytest.mark.parametrize("worker_cls", [RemoteSGLangEngine, RemotevLLMEngine])
def test_rollout_teacher_cleanup_only_drains_requested_worker_caches(
    storage: rt.HttpRTensorBackend, worker_cls: type
) -> None:
    """All workers receive nested input IDs without refetching or deleting storage."""
    scheduler = _LocalRPCScheduler(worker_cls)
    teacher = _teacher(scheduler)
    batch = rt.RTensor.remotize([_trajectory(), _trajectory()], node_addr="rollout")
    teacher.compute_logp(batch)
    source_stats = rt.storage_stats()
    fetches = storage.fetch.call_count
    for buffer in scheduler.buffers.values():
        assert buffer
        buffer["unrelated"] = torch.ones(1)

    teacher.clear_batches(batch, batch, None)

    assert rt.storage_stats() == source_stats
    assert storage.fetch.call_count == fetches
    assert all(set(buffer) == {"unrelated"} for buffer in scheduler.buffers.values())
    calls = [call for call in scheduler.calls if call[1] == "clear_batches"]
    assert len(calls) == 2
    for _, _, args, kwargs in calls:
        assert not args
        assert set(kwargs["shard_ids"]) == set(rt.flatten_shard_ids(batch))
        assert len(kwargs["shard_ids"]) == len(set(kwargs["shard_ids"]))
        assert kwargs["http_timeout"] == 10


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_rollout_teacher_cleanup_waits_for_other_workers_on_failure(
    storage: rt.HttpRTensorBackend, error_type: type[BaseException]
) -> None:
    """A failed worker must not cancel cleanup on slower peers or hide the error."""
    scheduler = _LocalRPCScheduler()
    teacher = _teacher(scheduler)
    batch = rt.RTensor.remotize([_trajectory(), _trajectory()], node_addr="rollout")
    teacher.compute_logp(batch)
    original_call = scheduler.async_call_engine
    failure = error_type("teacher cleanup failed")

    async def call(worker_id: str, *args: Any, **kwargs: Any) -> Any:
        if worker_id == "teacher/0":
            raise failure
        await asyncio.sleep(0.01)
        return await original_call(worker_id, *args, **kwargs)

    scheduler.async_call_engine = call
    with pytest.raises(error_type, match="worker teacher/0") as error:
        teacher.clear_batches(batch)

    if isinstance(failure, Exception):
        assert error.value is failure
    assert scheduler.buffers["teacher/0"]
    assert not scheduler.buffers["teacher/1"]


@pytest.mark.parametrize("spmd", [False, True])
def test_teacher_trainer_cleanup_respects_mode_and_actor_failure(
    monkeypatch: pytest.MonkeyPatch, spmd: bool
) -> None:
    """SPMD avoids remote cleanup; actor failures cannot skip a teacher drain."""
    monkeypatch.setenv("AREAL_SPMD_MODE", "1" if spmd else "0")
    teacher = Mock()
    teacher.compute_logp.return_value = [torch.zeros((1, 4))]
    trainer = _trainer(teacher)
    trainer.train_dataloader = [[{}]]
    batch = [_trajectory()]
    trainer.actor.prepare_batch.return_value = batch
    failure = RuntimeError("actor cleanup failed")
    trainer.actor.clear_batches.side_effect = failure

    if spmd:
        trainer.train(workflow=object())
        trainer.actor.clear_batches.assert_not_called()
        teacher.clear_batches.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="actor cleanup failed") as error:
            trainer.train(workflow=object())
        assert error.value is failure
        teacher.clear_batches.assert_called_once_with(batch)


def test_rollout_teacher_cleanup_without_remote_inputs_skips_rpc() -> None:
    """Local tensors and empty targets do not require cache eviction."""
    scheduler = _LocalRPCScheduler()
    teacher = _teacher(scheduler)

    teacher.clear_batches(None, [], {"local": torch.ones(1)})

    assert not scheduler.calls
