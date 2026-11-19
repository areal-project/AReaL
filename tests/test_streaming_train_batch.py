# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for streaming contracts, not distributed backend parity.

Keep configuration, policy state, admission, and trainer orchestration
separate. Tensor copies and PPO minibatch splitting are real; rollout/optimizer
services are small recording doubles. Backend parity lives in distributed tests.
"""

import asyncio
import threading
import time
import weakref
from collections import deque
from contextlib import nullcontext
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    DistillationConfig,
    FP8EngineConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    MicroBatchSpec,
    NormConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    TeacherConfig,
    TrainDatasetConfig,
)
from areal.api.engine_api import TrainEngine, _ParameterSnapshot
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher, WorkflowExecutor
from areal.trainer import rl_trainer
from areal.trainer.ppo.actor import PPOActor, PPOActorController, PPOActorControllerV2
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils import logging


# Configuration: use real dataclasses and only bypass worker initialization.
def _config(
    *, chunk_size=2, batch_size=8, n_samples=1, actor=None, rollout=None, **roles
) -> PPOConfig:
    return PPOConfig(
        experiment_name="test",
        trial_name="streaming",
        gconfig=GenerationHyperparameters(n_samples=n_samples),
        train_dataset=TrainDatasetConfig(batch_size=batch_size),
        rollout=rollout
        or InferenceEngineConfig(
            backend="vllm:d1",
            consumer_batch_size=batch_size,
            streaming_train_batch_size=chunk_size,
        ),
        actor=actor or PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=8),
        **roles,
    )


def _uninitialized_trainer(config: PPOConfig) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = config
    trainer.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
    trainer.rollout_alloc = ModelAllocation.from_str(
        config.rollout.backend, name="rollout"
    )
    for role in ("actor", "rollout", "critic", "ref", "teacher"):
        setattr(trainer, f"_should_offload_{role}", False)
    return trainer


@pytest.fixture
def single_controller(monkeypatch):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"chunk_size": 0}, "must be positive"),
        ({"chunk_size": 16}, "must not exceed"),
        ({"chunk_size": 3}, "train_dataset.batch_size must be divisible"),
        (
            {"actor": PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=6)},
            "actor.ppo_n_minibatches must divide or be divisible",
        ),
        (
            {
                "rollout": InferenceEngineConfig(
                    backend="vllm:d1",
                    consumer_batch_size=4,
                    streaming_train_batch_size=2,
                )
            },
            "consumer_batch_size to equal train_dataset.batch_size",
        ),
        ({"dynamic_bs": True}, "does not support dynamic_bs=true"),
        (
            {
                "rollout": InferenceEngineConfig(
                    backend="vllm:d1",
                    consumer_batch_size=8,
                    streaming_train_batch_size=2,
                    max_head_offpolicyness=1,
                )
            },
            "requires rollout.max_head_offpolicyness=0",
        ),
    ],
)
def test_streaming_configuration_rejects_invalid_batch_contracts(overrides, error):
    """Reject invalid streaming budgets at configuration construction, not training."""
    with pytest.raises(ValueError, match=error):
        _config(**overrides)


@pytest.mark.parametrize("n_minibatches", [1, 2, 4, 8])
def test_streaming_configuration_accepts_both_chunk_minibatch_relations(n_minibatches):
    _config(actor=PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=n_minibatches))


def test_streaming_keeps_uneven_minibatches_in_the_original_path():
    _config(
        batch_size=10,
        chunk_size=5,
        actor=PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=4),
    )


@pytest.mark.parametrize("n_minibatches", [0, -1])
def test_streaming_requires_positive_minibatch_count(n_minibatches):
    with pytest.raises(ValueError, match="must be positive"):
        _config(
            actor=PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=n_minibatches)
        )


@pytest.mark.parametrize(
    ("field", "norm"),
    [
        ("reward_norm", NormConfig(std_level=None)),
        ("adv_norm", NormConfig(mean_level=None)),
    ],
)
def test_batch_normalization_warning_describes_chunk_scope(field, norm):
    """Keep the documented chunk-local normalization semantics."""
    actor = PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=8, **{field: norm})
    with pytest.warns(UserWarning, match=rf"actor\.{field}") as warnings:
        _config(actor=actor)
    message = str(warnings[0].message)
    assert "streaming" in message and "full train batch" in message


@pytest.mark.parametrize(
    ("chunk_size", "recompute", "fp8_param", "prox_method", "reject"),
    [
        (2, True, True, None, True),
        (8, True, True, None, False),
        (None, True, True, None, False),
        (2, False, True, None, False),
        (2, True, False, None, False),
        (2, True, True, "loglinear", False),
        (2, False, True, "recompute", True),
        (2, True, True, "reuse", False),
    ],
)
def test_fp8_config_guard_only_applies_to_streaming_proximal_forward(
    single_controller, chunk_size, recompute, fp8_param, prox_method, reject
):
    """This config guard does not establish runtime quantized-storage safety."""
    actor = PPOActorConfig(
        backend="megatron:d1",
        ppo_n_minibatches=8,
        recompute_logprob=recompute,
    )
    actor.megatron.fp8_config = FP8EngineConfig(param=fp8_param)
    actor.megatron.ddp.fp8_param_gather = True
    if prox_method is not None:
        actor.use_decoupled_loss = True
        actor.prox_logp_method = prox_method
    trainer = _uninitialized_trainer(_config(chunk_size=chunk_size, actor=actor))
    if reject:
        with pytest.raises(ValueError, match="fp8_config.param=true"):
            trainer._validate_cfg()
    else:
        trainer._validate_cfg()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("colocate", [False, True])
def test_awex_streaming_requires_separated_actor(single_controller, version, colocate):
    config = _config(actor=PPOActorConfig(backend="megatron:d1", ppo_n_minibatches=8))
    config.actor._version = config.rollout._version = version
    config.actor.weight_update_mode = "awex"
    config.awex.enable_colocate_mode = colocate
    trainer = _uninitialized_trainer(config)
    if colocate and version == "v1":
        with pytest.raises(ValueError, match="colocated actor"):
            trainer._validate_cfg()
    else:
        trainer._validate_cfg()


@pytest.mark.parametrize(
    ("role", "n_samples"),
    [
        ("critic", 1),
        ("ref", 1),
        ("teacher", 1),
        ("critic", 2),
    ],
)
def test_forward_roles_require_atomic_prompt_groups_per_dp_rank(role, n_samples):
    """Response multiplicity must not split a prompt group across DP ranks."""
    role_config = PPOActorConfig(backend="fsdp:d4")
    if role == "critic":
        role_config = PPOCriticConfig(backend="fsdp:d4")
    elif role == "teacher":
        role_config = DistillationConfig(
            teachers=[TeacherConfig(engine_type="train", train=role_config)]
        )
    with pytest.raises(
        ValueError,
        match=rf"{'teacher-0' if role == 'teacher' else role} data-parallel size",
    ):
        _config(n_samples=n_samples, **{role: role_config})


@pytest.mark.parametrize(
    ("role", "mb_spec", "chunk_size", "batch_size", "error"),
    [
        ("ref", MicroBatchSpec(granularity=4), 2, 8, "ref.mb_spec.granularity"),
        ("ref", MicroBatchSpec(n_mbs=4), 2, 8, "ref.mb_spec"),
        ("critic", MicroBatchSpec(n_mbs=4), 2, 8, "critic.mb_spec"),
        (
            "teacher",
            MicroBatchSpec(n_mbs=5, n_mbs_divisor=4),
            6,
            12,
            "teacher-0.mb_spec",
        ),
    ],
)
def test_forward_microbatch_guard_uses_each_roles_effective_minimum(
    role, mb_spec, chunk_size, batch_size, error
):
    """Retain existing forward-role guards without broadening validation."""
    role_config = PPOActorConfig(backend="fsdp:d1", mb_spec=mb_spec)
    if role == "critic":
        role_config = PPOCriticConfig(backend="fsdp:d1", mb_spec=mb_spec)
    elif role == "teacher":
        role_config = DistillationConfig(
            teachers=[TeacherConfig(engine_type="train", train=role_config)]
        )
    with pytest.raises(ValueError, match=error):
        _config(chunk_size=chunk_size, batch_size=batch_size, **{role: role_config})


# Policy state: real CPU tensors, including exception exits and flat views.
def test_policy_state_enumeration_preserves_distinct_model_parts():
    """Expose each part's live parameters/persistent buffers, not forward caches."""
    parts = [torch.nn.Module(), torch.nn.Module()]
    for index, module in enumerate(parts):
        module.weight = torch.nn.Parameter(torch.tensor([float(index)]))
        module.register_buffer("policy_bias", torch.tensor([float(index + 10)]))
        module.register_buffer("cache", torch.tensor([0.0]), persistent=False)
    engine = SimpleNamespace(_get_policy_model_parts=lambda: parts)
    refs = list(TrainEngine._iter_policy_state_tensors(engine))
    expected = [
        tensor for module in parts for tensor in (module.weight, module.policy_bias)
    ]
    assert len(refs) == len(expected)
    assert len({key for key, _ in refs}) == len(expected)
    assert {id(tensor) for _, tensor in refs} == {id(tensor) for tensor in expected}
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    for module in parts:
        module.policy_bias.add_(10)
    with snapshot.activate(TrainEngine._iter_policy_state_tensors(engine)):
        for index, module in enumerate(parts):
            torch.testing.assert_close(
                module.policy_bias, torch.tensor([float(index + 10)]), rtol=0, atol=0
            )
            module.register_buffer("cache", torch.tensor([99.0]), persistent=False)
    for index, module in enumerate(parts):
        torch.testing.assert_close(
            module.policy_bias, torch.tensor([float(index + 20)]), rtol=0, atol=0
        )
        torch.testing.assert_close(module.cache, torch.tensor([99.0]), rtol=0, atol=0)
    snapshot.clear()


@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_snapshot_restores_flat_parameter_views_without_breaking_aliases(
    error_type, dtype
):
    """Restore shared flat storage, including frozen parameters and tied weights."""
    flat = torch.arange(10, dtype=dtype)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(flat[:4].view(2, 2))
    model.second = torch.nn.Parameter(flat[4:8].view(2, 2))
    model.frozen = torch.nn.Parameter(
        torch.tensor([2.0], dtype=dtype), requires_grad=False
    )
    model.tied = model.weight
    engine = SimpleNamespace(_get_policy_model_parts=lambda: [model])
    refs = list(TrainEngine._iter_policy_state_tensors(engine))
    assert len(refs) == 3
    storage = flat.untyped_storage().data_ptr()
    original_parameters = (model.weight, model.second, model.frozen)
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    with torch.no_grad():
        model.weight.add_(10)
        model.second.add_(20)
        model.frozen.add_(10)
        flat[8:].add_(30)
    current = flat.clone()
    context = (
        pytest.raises(error_type, match="forward failed")
        if error_type is not None
        else nullcontext()
    )
    with context:
        with snapshot.activate(TrainEngine._iter_policy_state_tensors(engine)):
            torch.testing.assert_close(
                flat[:8], torch.arange(8, dtype=dtype), rtol=0, atol=0
            )
            torch.testing.assert_close(flat[8:], current[8:], rtol=0, atol=0)
            torch.testing.assert_close(
                model.frozen, torch.tensor([2.0], dtype=dtype), rtol=0, atol=0
            )
            with torch.no_grad():
                model.weight.zero_()
                model.frozen.zero_()
            if error_type is not None:
                raise error_type("forward failed")
    torch.testing.assert_close(flat, current, rtol=0, atol=0)
    assert model.weight is original_parameters[0]
    assert model.second is original_parameters[1]
    assert model.frozen is original_parameters[2]
    assert model.tied is model.weight
    assert not model.frozen.requires_grad
    assert model.weight.untyped_storage().data_ptr() == storage
    assert model.second.untyped_storage().data_ptr() == storage
    torch.testing.assert_close(
        model.frozen, torch.tensor([12.0], dtype=dtype), rtol=0, atol=0
    )
    snapshot.clear()
    snapshot.clear()
    # A cleared snapshot can capture the next logical step's policy.
    snapshot.capture(TrainEngine._iter_policy_state_tensors(engine))
    with torch.no_grad():
        model.weight.add_(1)
    with snapshot.activate(TrainEngine._iter_policy_state_tensors(engine)):
        torch.testing.assert_close(model.weight, current[:4].view(2, 2), rtol=0, atol=0)
    torch.testing.assert_close(model.weight, current[:4].view(2, 2) + 1, rtol=0, atol=0)
    snapshot.clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_snapshot_cpu_backup_restores_latest_policy_without_retaining_backups(
    monkeypatch, dtype
):
    from areal.infra.platforms import current_platform

    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype))
    refs = [("weight", weight)]
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    old_policy = weight.detach().clone()
    storage = weight.untyped_storage().data_ptr()
    backups = []
    sync_states = []
    empty_like = torch.empty_like

    def record_backup(*args, **kwargs):
        backup = empty_like(*args, **kwargs)
        backups.append(weakref.ref(backup))
        return backup

    monkeypatch.setattr(torch, "empty_like", record_backup)
    monkeypatch.setattr(
        current_platform._ensure_initialized(),
        "synchronize",
        lambda: sync_states.append(snapshot.state),
    )
    for offset in (10, 20):
        current = old_policy + offset
        with torch.no_grad():
            weight.copy_(current)
        with snapshot.activate(iter(refs)):
            assert snapshot.state == "active"
            assert sync_states[-1] == "ready"
            assert backups[-1]().device.type == "cpu"
            assert backups[-1]().data_ptr() != storage
            assert (
                backups[-1]().data_ptr()
                != snapshot._snapshot_tensors_cpu[0][1].data_ptr()
            )
            torch.testing.assert_close(backups[-1](), current, rtol=0, atol=0)
            torch.testing.assert_close(weight, old_policy, rtol=0, atol=0)
            with torch.no_grad():
                weight.fill_(-1)
            torch.testing.assert_close(backups[-1](), current, rtol=0, atol=0)
            torch.testing.assert_close(
                snapshot._snapshot_tensors_cpu[0][1], old_policy, rtol=0, atol=0
            )
        torch.testing.assert_close(weight, current, rtol=0, atol=0)
        assert weight.untyped_storage().data_ptr() == storage
        assert snapshot.state == "ready"
        assert sync_states[-1] == "failed"
        assert all(backup() is None for backup in backups)
    assert len(backups) == 2
    snapshot.clear()


@pytest.mark.parametrize("failure", ["allocation", "copy"])
def test_snapshot_partial_cpu_backup_failure_syncs_without_overwriting_model_tensors(
    monkeypatch, failure
):
    from areal.infra.platforms import current_platform

    weights = [torch.tensor([1.0]), torch.tensor([2.0])]
    refs = list(zip(("first", "second"), weights, strict=True))
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    for weight in weights:
        weight.add_(10)
    current = [weight.clone() for weight in weights]
    events = []
    empty_like = torch.empty_like
    copy = torch.Tensor.copy_

    def allocate_backup(*args, **kwargs):
        events.append("allocate")
        if failure == "allocation" and events.count("allocate") == 2:
            raise RuntimeError("backup failed")
        return empty_like(*args, **kwargs)

    def copy_backup(destination, source, **kwargs):
        events.append("copy")
        if failure == "copy" and events.count("copy") == 2:
            raise RuntimeError("backup failed")
        return copy(destination, source, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty_like", allocate_backup)
        patch.setattr(torch.Tensor, "copy_", copy_backup)
        patch.setattr(
            current_platform._ensure_initialized(),
            "synchronize",
            lambda: events.append("sync"),
        )
        with pytest.raises(RuntimeError, match="backup failed"):
            with snapshot.activate(iter(refs)):
                pytest.fail("forward must not run after an incomplete backup")
    assert events.count("sync") == 1
    assert events[-1] == "sync"
    assert events.count("copy") == (1 if failure == "allocation" else 2)
    assert snapshot.state == "ready"
    for weight, expected in zip(weights, current, strict=True):
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    with snapshot.activate(iter(refs)):
        torch.testing.assert_close(weights[0], torch.tensor([1.0]), rtol=0, atol=0)
    for weight, expected in zip(weights, current, strict=True):
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    snapshot.clear()


def test_snapshot_cpu_restore_failure_keeps_failed_state(monkeypatch):
    weight = torch.tensor([1.0])
    refs = [("weight", weight)]
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    weight.add_(10)
    copy = torch.Tensor.copy_

    def fail_restore(destination, source, **kwargs):
        if snapshot.state == "failed":
            raise RuntimeError("restore failed")
        return copy(destination, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    with pytest.raises(RuntimeError, match="restore failed"):
        with snapshot.activate(iter(refs)):
            torch.testing.assert_close(weight, torch.tensor([1.0]), rtol=0, atol=0)
    assert snapshot.state == "failed"
    with pytest.raises(RuntimeError, match="ready policy snapshot"):
        with snapshot.activate(iter(refs)):
            pytest.fail("a failed restore must prevent another activation")
    snapshot.clear()


# Prefetch: exercise chunk returns with a synchronous task-runner double.
@pytest.fixture
def rejection_dispatcher():
    batch_size = 8
    manager = StalenessManager(
        version_provider=SimpleNamespace(get_version=lambda: 0),
        max_concurrent_rollouts=batch_size,
        consumer_batch_size=batch_size,
        max_staleness=0,
    )
    dispatcher = BatchTaskDispatcher.__new__(BatchTaskDispatcher)
    dispatcher.runner = SimpleNamespace(
        max_queue_size=2 * batch_size, get_input_queue_size=lambda: 0
    )
    dispatcher.staleness_manager = manager
    dispatcher._pending_inputs = deque()
    dispatcher._input_cv = threading.Condition()
    dispatcher.enable_tracing = False
    rejected = False

    def submit(item):
        dispatcher._pending_inputs.append(item)
        manager.on_rollout_enqueued()

    def wait(count, timeout):
        nonlocal rejected
        results = []
        for _ in range(count):
            item = dispatcher._pending_inputs.popleft()
            manager.on_rollout_submitted()
            if not rejected:
                rejected = True
                manager.on_rollout_rejected()
                results.append(None)
            else:
                manager.on_rollout_accepted()
                results.append(item)
        return results

    dispatcher.submit_task_input = submit
    dispatcher.wait_results = wait
    return dispatcher, iter(range(100))


def test_dispatcher_replaces_rejected_groups_across_chunk_returns(rejection_dispatcher):
    """A rejected result is replaced without losing or duplicating accepted groups."""
    dispatcher, inputs = rejection_dispatcher
    chunks = [
        dispatcher.active_submit_and_wait(inputs, batch_size=2, prefetch_size=8)
        for _ in range(4)
    ]
    assert all(len(chunk) == 2 for chunk in chunks)
    assert [item for chunk in chunks for item in chunk] == list(range(1, 9))


def test_threaded_dispatcher_returns_complete_groups_while_remaining_tasks_run(
    monkeypatch,
):
    """Use real runner/conditions; later groups stay in flight after the first chunk."""
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    manager = StalenessManager(
        version_provider=SimpleNamespace(get_version=lambda: 0),
        max_concurrent_rollouts=4,
        consumer_batch_size=4,
        max_staleness=0,
    )

    def task_factory(task_input):
        async def run():
            if task_input.task_id >= 2:
                started[task_input.task_id - 2].set()
                await asyncio.to_thread(release.wait)
            manager.on_rollout_accepted()
            return {"input_ids": torch.full((2, 3), task_input.task_id)}

        return run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=8, task_factory=task_factory, staleness_manager=manager
    )
    deadline = time.monotonic() + 10
    wait_results = dispatcher.wait_results

    def bounded_wait(*args, **kwargs):
        assert time.monotonic() < deadline, (
            "streaming dispatcher stopped making progress"
        )
        return wait_results(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "wait_results", bounded_wait)
    inputs = (SimpleNamespace(task_id=index) for index in count())
    try:
        dispatcher.initialize(logging.getLogger("StreamingDispatcherTest"))
        first = dispatcher.active_submit_and_wait(inputs, batch_size=2, prefetch_size=4)
        assert sorted(int(group["input_ids"][0, 0]) for group in first) == [0, 1]
        assert all(event.wait(timeout=5) for event in started)
        assert not release.is_set()
        release.set()
        last = dispatcher.active_submit_and_wait(inputs, batch_size=2, prefetch_size=4)
        assert sorted(int(group["input_ids"][0, 0]) for group in last) == [2, 3]
        for group in first + last:
            assert group["input_ids"].shape == (2, 3)
            torch.testing.assert_close(
                group["input_ids"],
                group["input_ids"][0, 0].expand(2, 3),
                rtol=0,
                atol=0,
            )
    finally:
        release.set()
        dispatcher.destroy()


@pytest.mark.parametrize(
    "controller_type",
    [RolloutController, WorkflowExecutor],
    ids=["v1", "v2-executor"],
)
@pytest.mark.parametrize("chunk_size", [None, 2], ids=["full-batch", "streaming"])
def test_prepare_batch_passes_chunk_return_size_and_full_prefetch(
    controller_type, chunk_size
):
    """Exercise the real v1/v2 prepare methods, mocking only dispatcher submission."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    config = _config(chunk_size=chunk_size).rollout
    if controller_type is RolloutController:
        controller = RolloutController(
            inf_engine=Mock(), config=config, scheduler=Mock()
        )
    else:
        controller = WorkflowExecutor(config=config, inference_engine=Mock())
    trajectory = {"input_ids": torch.ones((2, 3), dtype=torch.long)}
    batch_size = chunk_size or 8
    controller._dispatcher = Mock()
    controller._dispatcher.active_submit_and_wait.return_value = [
        SimpleNamespace(trajectory=trajectory) for _ in range(batch_size)
    ]
    dataloader = StatefulDataLoader(
        [{"prompt": str(index)} for index in range(8)],
        batch_size=8,
        collate_fn=lambda batch: batch,
    )
    results = controller.prepare_batch(dataloader=dataloader, workflow=None)
    assert len(results) == batch_size
    assert all(result is trajectory for result in results)
    kwargs = controller._dispatcher.active_submit_and_wait.call_args.kwargs
    assert kwargs["batch_size"] == batch_size
    assert kwargs["prefetch_size"] == 8


# Orchestration: record service boundaries, while snapshots still copy tensors.
@pytest.fixture
def quiet_instrumentation(monkeypatch, single_controller):
    for name in ("record_timing", "scope"):
        monkeypatch.setattr(
            rl_trainer.stats_tracker, name, lambda *_args, **_kwargs: nullcontext()
        )
    for name in ("scalar", "stat", "stat_compact", "denominator"):
        monkeypatch.setattr(
            rl_trainer.stats_tracker, name, lambda *_args, **_kwargs: None
        )
    monkeypatch.setattr(
        rl_trainer.perf_tracer, "trace_scope", lambda *_args, **_kwargs: nullcontext()
    )


class _RecordingActor:
    def __init__(self, events, chunk_size):
        self.events = events
        self.chunk_size = chunk_size
        self.chunk = self.version = 0
        self.weight = torch.tensor([0.0])
        self.snapshot = _ParameterSnapshot()
        self.updates = []
        self.accumulation_calls = []
        self.logps = []
        self.cleared_batches = []
        self._exported_updates = 0

    def prepare_batch(self, *_args, **_kwargs):
        self.chunk += 1
        self.events.append(("prepare", self.chunk, self.version))
        start = (self.chunk - 1) * self.chunk_size
        return [
            {"chunk": self.chunk, "sample": i}
            for i in range(start, start + self.chunk_size)
        ]

    def compute_advantages(self, batch):
        return batch

    def capture_rollout_policy_snapshot(self):
        self.snapshot.capture(iter([("weight", self.weight)]))
        self.events.append(("capture",))

    def compute_logp(self, batch, *, use_rollout_policy_snapshot=False):
        context = (
            self.snapshot.activate(iter([("weight", self.weight)]))
            if use_rollout_policy_snapshot
            else nullcontext()
        )
        with context:
            value = self.weight.item()
            self.logps.append((self.chunk, value, use_rollout_policy_snapshot))
            return [value] * len(batch)

    def clear_rollout_policy_snapshot(self):
        self.snapshot.clear()
        self.events.append(("snapshot_clear",))

    def ppo_update(
        self, batch, *, n_minibatches=8, accumulation_steps=1, accumulation_index=0
    ):
        self.updates.append(([item["sample"] for item in batch], n_minibatches))
        self.accumulation_calls.append((accumulation_steps, accumulation_index))
        self.events.append(("update", self.chunk))
        if accumulation_index == accumulation_steps - 1:
            self.weight.add_(1)

    def update_weights(self, meta):
        self.events.append(("publish", meta.version, len(self.updates)))

    def step_lr_scheduler(self):
        self.events.append(("actor_lr",))

    def clear_batches(self, *batches):
        self.cleared_batches.append(
            tuple(tuple(item["sample"] for item in batch) for batch in batches)
        )
        self.events.append(("actor_clear",))

    def set_version(self, version):
        self.version = version

    def clear_all_local_rtensors(self):
        pass

    def get_device_stats(self):
        return SimpleNamespace(log=lambda _message: None)

    def export_stats(self):
        count = len(self.updates) - self._exported_updates
        self._exported_updates = len(self.updates)
        return {
            "timeperf/rollout": 1.5,
            "timeperf/rollout__count": count,
            "timeperf/train_step": 2.0,
            "timeperf/train_step__count": count,
        }


class _RecordingCritic:
    def __init__(self, events):
        self.events = events
        self.cleared_batches = []

    def compute_values(self, batch):
        self.events.append(("values", batch[0]["chunk"]))
        return [0] * len(batch)

    def ppo_update(self, batch):
        self.events.append(("critic_update", tuple(item["sample"] for item in batch)))

    def step_lr_scheduler(self):
        self.events.append(("critic_lr",))

    def clear_batches(self, *batches):
        self.cleared_batches.append(
            tuple(tuple(item["sample"] for item in batch) for batch in batches)
        )
        self.events.append(("critic_clear",))

    def set_version(self, version):
        self.version = version

    def clear_all_local_rtensors(self):
        pass

    def get_device_stats(self):
        return SimpleNamespace(log=lambda _message: None)

    def onload(self):
        self.events.append(("critic_onload",))

    def offload(self):
        self.events.append(("critic_offload",))


@pytest.fixture
def trainer_factory(quiet_instrumentation):
    def make(
        *,
        chunk_size=2,
        critic=False,
        critic_offload=False,
        recompute=False,
        steps=1,
        n_minibatches=8,
    ):
        config = _config(
            chunk_size=chunk_size,
            actor=PPOActorConfig(
                backend="fsdp:d1",
                ppo_n_minibatches=n_minibatches,
                recompute_logprob=recompute,
            ),
            critic=(
                PPOCriticConfig(backend="fsdp:d1", offload=critic_offload)
                if critic
                else None
            ),
        )
        config.total_train_epochs = 1
        config.total_train_steps = steps
        config.actor.weight_update_mode = "xccl"
        trainer = _uninitialized_trainer(config)
        events = []
        trainer.recover_info = None
        trainer.train_dataloader = [None] * steps
        trainer.actor = _RecordingActor(events, chunk_size or 8)
        trainer.critic = _RecordingCritic(events) if critic else None
        trainer.ref = trainer.eval_rollout = trainer.data_controller = None
        trainer.teachers = []
        trainer.teacher_configs = []
        trainer._should_offload_critic = critic_offload
        trainer.config.enable_offload = critic_offload
        trainer._validate_cfg()
        trainer.rollout = SimpleNamespace(
            pause=Mock(side_effect=lambda: events.append(("pause",))),
            resume=Mock(),
            set_version=Mock(),
            export_stats=lambda: {},
        )
        trainer.stats_logger = SimpleNamespace(commit=Mock())
        trainer.saver = SimpleNamespace(maybe_wait_for_staging=Mock())
        trainer.weight_update_meta = SimpleNamespace(
            with_version=lambda version: SimpleNamespace(
                version=version, colocate_mode=False
            )
        )
        trainer._requires_proxy_workflow = lambda _workflow, _kwargs=None: False
        trainer._save_hf = Mock()
        trainer._save_recover_checkpoint = Mock()
        trainer._evaluate = Mock()
        trainer._save_perf_tracer = Mock()

        return trainer

    return make


def test_each_chunk_updates_actor_before_the_next_chunk_is_consumed(
    trainer_factory,
):
    """Protect streaming interleaving, sample ownership, and minibatch budgets."""
    chunk_size = 2
    trainer = trainer_factory(chunk_size=chunk_size)
    trainer.train(workflow=object())
    chunks = 8 // chunk_size
    boundaries = [
        event for event in trainer.actor.events if event[0] in {"prepare", "update"}
    ]
    assert boundaries == [
        event
        for chunk in range(1, chunks + 1)
        for event in (("prepare", chunk, 0), ("update", chunk))
    ]
    assert trainer.actor.updates == [
        (list(range(start, start + chunk_size)), chunk_size)
        for start in range(0, 8, chunk_size)
    ]
    assert trainer.actor.events.count(("actor_clear",)) == 1
    assert trainer.actor.cleared_batches == [(tuple(range(8)), tuple(range(8)))]


@pytest.mark.parametrize("chunk_size", [None, 8])
def test_single_chunk_retains_full_batch_trainer_calls(trainer_factory, chunk_size):
    trainer = trainer_factory(chunk_size=chunk_size, recompute=True, critic=True)
    trainer.train(workflow=object())
    assert trainer.actor.updates == [(list(range(8)), 8)]
    assert trainer.actor.accumulation_calls == [(1, 0)]
    assert trainer.actor.logps == [(1, 0.0, False)]
    assert not any(
        event[0] in {"capture", "snapshot_clear"} for event in trainer.actor.events
    )
    assert trainer.actor.events.count(("actor_lr",)) == 1
    assert trainer.actor.events.count(("critic_lr",)) == 1
    assert trainer.stats_logger.commit.call_args.args[3]["timeperf/train_step"] == 2.0


@pytest.mark.parametrize("n_minibatches", [1, 2])
def test_chunks_accumulate_until_each_minibatch_boundary(
    trainer_factory, n_minibatches
):
    trainer = trainer_factory(n_minibatches=n_minibatches, critic=True, recompute=True)
    trainer.train(workflow=object())
    accumulation_steps = 4 // n_minibatches
    assert trainer.actor.accumulation_calls == [
        (accumulation_steps, index % accumulation_steps) for index in range(4)
    ]
    assert all(count == 1 for _, count in trainer.actor.updates)
    torch.testing.assert_close(
        trainer.actor.weight, torch.tensor([float(n_minibatches)]), rtol=0, atol=0
    )
    assert trainer.actor.events.count(("actor_lr",)) == 1
    assert trainer.actor.events.count(("critic_lr",)) == 1
    assert ("critic_update", tuple(range(8))) in trainer.actor.events
    assert len([e for e in trainer.actor.events if e[0] == "publish"]) == 1
    assert trainer.actor.cleared_batches == [(tuple(range(8)), tuple(range(8)))]
    assert all(value == 0 for _, value, _ in trainer.actor.logps)


@pytest.mark.parametrize("n_minibatches", [1, 2, 8])
def test_lr_and_weight_publication_advance_only_at_logical_batch_boundaries(
    trainer_factory,
    n_minibatches,
):
    """No mid-batch LR/version advance; retain target rollout resume behavior."""
    steps = 2
    trainer = trainer_factory(steps=steps, n_minibatches=n_minibatches)
    trainer.train(workflow=object())
    events = trainer.actor.events
    assert [event for event in events if event[0] == "publish"] == [
        ("publish", step, step * 4) for step in range(1, steps + 1)
    ]
    lr_positions = [i for i, event in enumerate(events) if event == ("actor_lr",)]
    pause_positions = [i for i, event in enumerate(events) if event == ("pause",)]
    assert len(lr_positions) == len(pause_positions) == steps
    for step, (lr, pause) in enumerate(zip(lr_positions, pause_positions), start=1):
        publish = events.index(("publish", step, step * 4))
        last_update = events.index(("update", step * 4))
        assert last_update < lr < publish
        assert last_update < pause < publish
        if step < steps:
            assert publish < events.index(("prepare", step * 4 + 1, step))
    assert [event[2] for event in events if event[0] == "prepare"] == [
        version for version in range(steps) for _ in range(4)
    ]
    assert trainer.rollout.resume.call_count == steps
    assert trainer._save_recover_checkpoint.call_count == steps
    assert trainer._evaluate.call_count == steps
    assert trainer.saver.maybe_wait_for_staging.call_count == steps
    for call in trainer.stats_logger.commit.call_args_list:
        stats = call.args[3]
        assert stats["timeperf/rollout"] == 6.0
        assert stats["timeperf/train_step"] == 8.0


@pytest.mark.parametrize(
    "recompute", [False, True], ids=["skip-forward", "fixed-policy-forward"]
)
@pytest.mark.parametrize("n_minibatches", [1, 2, 4, 8])
def test_proximal_forward_uses_fixed_weights_without_losing_actor_updates(
    trainer_factory, recompute, n_minibatches, monkeypatch
):
    """Use live weights before the first update, then the fixed policy snapshot."""
    steps = 2
    trainer = trainer_factory(
        recompute=recompute, steps=steps, n_minibatches=n_minibatches
    )
    actor = trainer.actor
    needs_snapshot = recompute and n_minibatches > 1
    if not needs_snapshot:

        def unexpected_snapshot(*_args, **_kwargs):
            pytest.fail("this path must not copy or activate a policy snapshot")

        monkeypatch.setattr(
            actor, "capture_rollout_policy_snapshot", unexpected_snapshot
        )
        monkeypatch.setattr(actor, "clear_rollout_policy_snapshot", unexpected_snapshot)
        monkeypatch.setattr(actor.snapshot, "activate", unexpected_snapshot)
    trainer.train(workflow=object())
    # This recording actor changes once per completed call/window.
    updates_per_step = min(n_minibatches, 4)
    chunks_before_first_update = max(1, 4 // n_minibatches)
    if recompute:
        assert actor.logps == [
            (
                step * 4 + chunk,
                float(step * updates_per_step),
                needs_snapshot and chunk > chunks_before_first_update,
            )
            for step in range(steps)
            for chunk in range(1, 5)
        ]
    else:
        assert actor.logps == []
    if needs_snapshot:
        assert actor.events.count(("capture",)) == steps
        assert actor.events.count(("snapshot_clear",)) == steps
        captures = [i for i, event in enumerate(actor.events) if event == ("capture",)]
        clears = [
            i for i, event in enumerate(actor.events) if event == ("snapshot_clear",)
        ]
        for step in range(steps):
            assert captures[step] < actor.events.index(("update", step * 4 + 1))
            assert actor.events.index(("update", step * 4 + 3)) < clears[step]
            assert clears[step] < actor.events.index(("update", step * 4 + 4))
    else:
        assert not any(
            event[0] in {"capture", "snapshot_clear"} for event in actor.events
        )
    torch.testing.assert_close(
        actor.weight, torch.tensor([float(steps * updates_per_step)]), rtol=0, atol=0
    )


@pytest.mark.parametrize("offload", [False, True])
def test_critic_update_and_batch_cleanup_wait_until_all_actor_chunks_finish(
    trainer_factory, offload
):
    """All values are read before one full-batch critic update and cache release."""
    trainer = trainer_factory(critic=True, critic_offload=offload, recompute=True)
    trainer.train(workflow=object())
    events = trainer.actor.events
    critic_update = ("critic_update", tuple(range(8)))
    assert events.count(critic_update) == 1
    assert [event for event in events if event[0] == "values"] == [
        ("values", chunk) for chunk in range(1, 5)
    ]
    assert all(
        index < events.index(critic_update)
        for index, event in enumerate(events)
        if event[0] == "values"
    )
    assert trainer.actor.logps == [(chunk, 0.0, chunk > 1) for chunk in range(1, 5)]
    assert events.index(("update", 4)) < events.index(critic_update)
    assert events.index(critic_update) < events.index(("pause",))
    assert events.count(("critic_lr",)) == 1
    for role in ("actor", "critic"):
        assert events.count((f"{role}_clear",)) == 1
        assert events.index(critic_update) < events.index((f"{role}_clear",))
    assert trainer.actor.cleared_batches == [(tuple(range(8)), tuple(range(8)))]
    assert trainer.critic.cleared_batches == [(tuple(range(8)), tuple(range(8)))]
    if offload:
        assert events.count(("critic_onload",)) == 1
        assert events.count(("critic_offload",)) == 1
        assert events.index(("critic_onload",)) < events.index(("values", 1))
        assert events.index(("critic_lr",)) < events.index(("critic_offload",))
        assert events.index(("critic_offload",)) < events.index(("pause",))
    else:
        assert not any(
            event[0] in {"critic_onload", "critic_offload"} for event in events
        )


# Worker boundary: run real PPO concatenation/splitting, fake only engine updates.
@pytest.mark.parametrize("use_snapshot", [False, True])
def test_real_actor_compute_logp_selects_snapshot_and_restores_live_policy(
    use_snapshot,
):
    """The actual actor forward uses old weights only when requested."""
    weight = torch.tensor([1.0])
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter([("weight", weight)]))
    weight.add_(10)
    engine = SimpleNamespace(
        eval=Mock(),
        parameter_snapshot_context=lambda: snapshot.activate(
            iter([("weight", weight)])
        ),
        forward=lambda input_, **_kwargs: weight
        * torch.ones_like(input_["input_ids"], dtype=torch.float32),
    )
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1"), engine)
    groups = [
        {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
        }
        for shape in ((1, 4), (2, 3))
    ]
    logps = actor.compute_logp(groups, use_rollout_policy_snapshot=use_snapshot)
    for logp, group in zip(logps, groups, strict=True):
        torch.testing.assert_close(
            logp,
            torch.full(group["input_ids"].shape, 1.0 if use_snapshot else 11.0),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(weight, torch.tensor([11.0]), rtol=0, atol=0)
    snapshot.clear()


@pytest.mark.parametrize(
    "controller_type", [PPOActorController, PPOActorControllerV2], ids=["v1", "v2"]
)
def test_actor_controllers_forward_snapshot_lifecycle_and_logp_flag(controller_type):
    """Check client RPC contracts, not HTTP routing or distributed communication."""
    controller = controller_type.__new__(controller_type)
    results = [torch.tensor([[1.0, 2.0]])]
    data = [{"input_ids": torch.tensor([[1, 2]])}]
    if controller_type is PPOActorController:
        controller._custom_function_call = Mock(return_value=results)
    else:
        controller._gateway_post = Mock()
        controller._gateway_post_result = Mock(return_value=results)
    controller.capture_rollout_policy_snapshot()
    assert controller.compute_logp(data, use_rollout_policy_snapshot=True) is results
    controller.clear_rollout_policy_snapshot()
    if controller_type is PPOActorController:
        calls = controller._custom_function_call.call_args_list
        assert [call.args[0] for call in calls] == [
            "capture_parameter_snapshot",
            "compute_logp",
            "clear_parameter_snapshot",
        ]
        assert all(call.kwargs["rpc_meta"] == {"broadcast": True} for call in calls)
        assert calls[1].args[1] is data
        assert calls[1].kwargs["use_rollout_policy_snapshot"] is True
    else:
        assert controller._gateway_post.call_args_list[0].args == (
            "/ppo/actor/capture_rollout_policy_snapshot",
            {},
        )
        assert controller._gateway_post.call_args_list[1].args == (
            "/ppo/actor/clear_rollout_policy_snapshot",
            {},
        )
        path, payload = controller._gateway_post_result.call_args.args
        assert path == "/ppo/actor/compute_logp"
        assert deserialize_value(payload["kwargs"])["use_rollout_policy_snapshot"]
        restored = deserialize_value(payload["args"])[0]
        torch.testing.assert_close(
            restored[0]["input_ids"], data[0]["input_ids"], rtol=0, atol=0
        )


@pytest.mark.parametrize("override", [False, True], ids=["default", "chunk-override"])
@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_real_actor_preserves_complete_minibatches_inside_each_chunk(
    quiet_instrumentation, override, accumulation_steps
):
    """Each optimizer call receives complete sequences from only one chunk."""
    minibatches = []
    accumulation_calls = []

    def record_update(batch, **_kwargs):
        minibatches.append(batch["input_ids"].clone())
        accumulation_calls.append(
            {
                key: value
                for key, value in _kwargs.items()
                if key.startswith("accumulation_")
            }
        )
        return {}

    engine = SimpleNamespace(
        train=Mock(),
        data_parallel_group=None,
        device="cpu",
        get_version=lambda: 0,
        train_batch=Mock(side_effect=record_update),
    )
    actor = PPOActor(
        PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=4 if override else 2),
        engine,
    )
    for start in (0, 4):
        trajectories = [
            {
                "input_ids": torch.full((1, 4), sample, dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.bool),
                "loss_mask": torch.tensor([[False, True, True, False]]),
                "rewards": torch.tensor([1.0]),
                "advantages": torch.ones((1, 4)),
                "kl_rewards": torch.zeros((1, 4)),
                "tot_rewards": torch.ones((1, 4)),
                "is_truncated": torch.tensor([False]),
            }
            for sample in range(start, start + 4)
        ]
        kwargs = {"n_minibatches": 2} if override else {}
        if accumulation_steps > 1:
            kwargs = dict(
                n_minibatches=1,
                accumulation_steps=accumulation_steps,
                accumulation_index=start // 4,
            )
        actor.ppo_update(trajectories, **kwargs)
        count = 1 if accumulation_steps > 1 else 2
        chunk_minibatches = minibatches[-count:]
        assert len(minibatches) == (start // 4 + 1) * count
        assert [tuple(batch.shape) for batch in chunk_minibatches] == [
            (4 // count, 4)
        ] * count
        assert accumulation_calls[-1] == (
            dict(accumulation_steps=2, accumulation_index=start // 4)
            if accumulation_steps > 1
            else {}
        )
        rows = torch.cat(chunk_minibatches)
        rows = rows[rows[:, 0].argsort()]
        expected = torch.arange(start, start + 4)[:, None].expand(4, 4)
        torch.testing.assert_close(rows, expected, rtol=0, atol=0)


def test_actor_rejects_multiple_minibatches_per_accumulated_chunk():
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1", ppo_n_minibatches=2), Mock())
    with pytest.raises(ValueError, match="requires n_minibatches=1"):
        actor.ppo_update([], accumulation_steps=2)


@pytest.mark.parametrize("controller_type", [PPOActorController, PPOActorControllerV2])
def test_actor_controllers_forward_accumulation_window(controller_type):
    controller = controller_type.__new__(controller_type)
    if controller_type is PPOActorController:
        controller._custom_function_call = Mock()
    else:
        controller._gateway_post = Mock()
    controller.ppo_update(
        [], n_minibatches=1, accumulation_steps=4, accumulation_index=2
    )
    if controller_type is PPOActorController:
        kwargs = controller._custom_function_call.call_args.kwargs
    else:
        _, payload = controller._gateway_post.call_args.args
        kwargs = deserialize_value(payload["kwargs"])
    assert kwargs["accumulation_steps"] == 4
    assert kwargs["accumulation_index"] == 2
