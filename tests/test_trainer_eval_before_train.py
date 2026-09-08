# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from weakref import ref

import pytest

from areal.trainer import dpo_trainer, rl_trainer, rw_trainer, sft_trainer
from areal.trainer.dpo_trainer import DPOTrainer
from areal.trainer.rl_trainer import PPOTrainer
from areal.trainer.rw_trainer import RWTrainer
from areal.trainer.sft_trainer import SFTTrainer
from areal.utils import stats_logger as stats_logger_module
from areal.utils.stats_logger import StatsLogger


class _StopAfterFirstUpdate(Exception):
    pass


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_ppo_evaluation_releases_v1_results_after_each_evaluation(monkeypatch, version):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version=version),
        eval_gconfig=SimpleNamespace(n_samples=2),
    )
    trainer.valid_dataloader = [[{"id": 1}], [{"id": 2}]]
    results = [[{"remote_shard": object()}], [None, {"remote_shard": object()}]]
    pending = iter(results)
    cleaned = []
    submitted = []
    trainer.actor = SimpleNamespace(
        is_data_parallel_head=lambda: True,
        clear_batches=cleaned.append,
    )

    def wait(count, timeout):
        assert count == 2 and timeout is None
        return next(pending)

    trainer.eval_rollout = SimpleNamespace(
        submit=lambda *args, **kwargs: submitted.append((args, kwargs)),
        wait=wait,
    )
    for _ in results:
        trainer._evaluate_fn("eval-workflow", {})
    assert cleaned == (results if version == "v1" else [])
    assert len(submitted) == 4
    assert all(kwargs["is_eval"] for _, kwargs in submitted)


def test_ppo_evaluation_does_not_hide_result_cleanup_failure(monkeypatch):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version="v1"),
        eval_gconfig=SimpleNamespace(n_samples=1),
    )
    trainer.valid_dataloader = [[{}]]

    def fail_cleanup(_results):
        raise RuntimeError("storage deletion failed")

    trainer.actor = SimpleNamespace(
        is_data_parallel_head=lambda: True, clear_batches=fail_cleanup
    )
    trainer.eval_rollout = SimpleNamespace(
        submit=lambda *args, **kwargs: None,
        wait=lambda *args, **kwargs: [{}],
    )
    with pytest.raises(RuntimeError, match="storage deletion failed"):
        trainer._evaluate_fn("eval-workflow", {})


class _FakeSaver:
    def maybe_wait_for_staging(self) -> None:
        pass


class _FakeLastStepInfo:
    def next(self):
        return SimpleNamespace(global_step=1)


class _FakeDeviceStats:
    def log(self, _message: str) -> None:
        pass


class _FakeLogprobs:
    ndim = 1

    def unsqueeze(self, _dim: int):
        return self


class _FakeRef:
    def compute_logp(self, batch):
        return [_FakeLogprobs() for _ in batch]

    def get_device_stats(self) -> _FakeDeviceStats:
        return _FakeDeviceStats()


class _CallbackEvaluator:
    def evaluate_before_train(self, evaluate_fn) -> bool:
        evaluate_fn()
        return True


class _RecordingEvaluator:
    def __init__(self):
        self.callbacks = []

    def evaluate_before_train(self, evaluate_fn) -> bool:
        self.callbacks.append(evaluate_fn)
        return False


class _FailingUpdateActor:
    def __init__(self, update_method: str, events: list[tuple]):
        self.update_method = update_method
        self.events = events

    def __getattr__(self, name: str):
        if name == self.update_method:
            return self._update
        raise AttributeError(name)

    def _update(self, _batch) -> None:
        self.events.append(("update", {}))
        raise _StopAfterFirstUpdate


class _FailingPPOActor:
    def __init__(self, events: list[tuple]):
        self.events = events

    def prepare_batch(self, *_args, **_kwargs):
        return [{}]

    def compute_advantages(self, _rollout_batch):
        return [{}]

    def get_device_stats(self) -> _FakeDeviceStats:
        return _FakeDeviceStats()

    def ppo_update(self, _adv_batch) -> None:
        self.events.append(("update", {}))
        raise _StopAfterFirstUpdate


def _disable_timing_contexts(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module.stats_tracker,
        "record_timing",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        module.perf_tracer,
        "trace_scope",
        lambda *_args, **_kwargs: nullcontext(),
    )


def _record_initial_eval(events: list[tuple], *_args, **_kwargs) -> bool:
    events.append(("initial_eval", {}))
    return True


def _record_commit(events: list[tuple], **kwargs) -> None:
    events.append(("commit", kwargs))


def _build_supervised_trainer(
    trainer_cls,
    update_method: str,
    events: list[tuple],
    *,
    recovered: bool = False,
):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = SimpleNamespace(
        total_train_epochs=1,
        total_train_steps=None,
        memory_profiler=None,
    )
    trainer.recover_info = (
        SimpleNamespace(last_step_info=_FakeLastStepInfo()) if recovered else None
    )
    trainer.train_dataloader = [[{}], [{}]] if recovered else [[{}]]
    trainer.saver = _FakeSaver()
    trainer.actor = _FailingUpdateActor(update_method, events)
    trainer._load_bcast_from = lambda data_generator: next(data_generator)
    trainer._evaluate_before_train = lambda: _record_initial_eval(events)
    trainer._export_and_commit_stats = lambda **kwargs: _record_commit(events, **kwargs)
    if trainer_cls is DPOTrainer:
        trainer.ref = _FakeRef()
    return trainer


def _build_ppo_trainer(events: list[tuple], *, recovered: bool = False):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        total_train_epochs=1,
        total_train_steps=None,
        rollout=SimpleNamespace(agent=None),
        gconfig=SimpleNamespace(
            n_samples=1,
            reward_normalization=False,
            drop_incomplete_group=False,
        ),
        dynamic_bs=False,
        actor=SimpleNamespace(
            _version="v1",
            weight_update_mode="xccl",
            should_compute_prox_logp=lambda: False,
        ),
        memory_profiler=None,
    )
    trainer.recover_info = (
        SimpleNamespace(last_step_info=_FakeLastStepInfo()) if recovered else None
    )
    trainer.train_dataloader = [[{}], [{}]] if recovered else [[{}]]
    trainer.saver = _FakeSaver()
    trainer.actor = _FailingPPOActor(events)
    trainer.critic = None
    trainer.ref = None
    trainer.teacher = None
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._requires_proxy_workflow = lambda _workflow: False
    trainer._evaluate_before_train = lambda *_args, **_kwargs: _record_initial_eval(
        events
    )
    trainer._export_and_commit_stats = lambda **kwargs: _record_commit(events, **kwargs)
    return trainer


def test_awex_reference_scoring_releases_rollout_before_restoring_models(monkeypatch):
    """A shared GPU belongs to rollout, then reference, then the PPO actor."""
    _disable_timing_contexts(monkeypatch, rl_trainer)
    events = []
    trainer = _build_ppo_trainer(events)
    trainer.config.actor.weight_update_mode = "awex"
    trainer.mopd_execution_plan = None
    trainer.mopd_teacher_phase = None
    trainer._should_offload_ref = True
    trainer.actor.onload = lambda: events.append(("actor_onload", {}))
    trainer.rollout = SimpleNamespace(
        pause=lambda: events.append(("rollout_pause", {})),
        pause_generation_sync=lambda: events.append(("generation_pause", {})),
        offload=lambda tags: events.append(("rollout_offload", {"tags": tags})),
    )

    def score(batch):
        events.append(("ref_score", {}))
        return [0.0 for _ in batch]

    trainer.ref = SimpleNamespace(
        onload=lambda: events.append(("ref_onload", {})),
        compute_logp=score,
        offload=lambda: events.append(("ref_offload", {})),
        get_device_stats=lambda: _FakeDeviceStats(),
    )
    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train(workflow="test-workflow")
    phase_events = [
        event for event, _ in events if event not in ("initial_eval", "commit")
    ]
    assert phase_events == [
        "rollout_pause",
        "generation_pause",
        "rollout_offload",
        "rollout_offload",
        "rollout_offload",
        "ref_onload",
        "ref_score",
        "ref_offload",
        "actor_onload",
        "update",
    ]
    assert [
        details["tags"] for event, details in events if event == "rollout_offload"
    ] == [
        ["kv_cache"],
        ["weights"],
        ["cuda_graph"],
    ]


@pytest.mark.parametrize("mode", ["awex", "xccl"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_ppo_releases_training_payloads_before_save_and_eval(
    monkeypatch, mode, cleanup_fails
):
    _disable_timing_contexts(monkeypatch, rl_trainer)
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    events, cache, images = [], {}, []
    trainer = _build_ppo_trainer(events)
    trainer.config.actor.weight_update_mode = mode
    trainer.mopd_execution_plan = trainer.mopd_teacher_phase = None
    trainer._should_offload_ref = trainer._should_offload_critic = False
    trainer._evaluate_before_train = lambda **kwargs: False

    class ImagePayload:
        pass

    def prepare(*args, **kwargs):
        payload = ImagePayload()
        images.append(ref(payload))
        batch = [{"multi_modal_input": payload}]
        cache.update(actor=batch, ref=batch, critic=batch, data=batch)
        return batch

    def clear(role, *args):
        events.append(("clear_" + role, {}))
        cache.pop(role)
        if cleanup_fails and role == "actor":
            raise RuntimeError("actor cleanup failed")

    def after_release(phase, **kwargs):
        assert not cache
        assert images[0]() is None
        events.append((phase, {}))

    trainer.actor.prepare_batch = prepare
    trainer.actor.compute_advantages = lambda batch: batch
    trainer.actor.ppo_update = lambda batch: events.append(("actor_update", {}))
    trainer.actor.step_lr_scheduler = lambda: None
    trainer.actor.onload = lambda: None
    trainer.actor.clear_batches = lambda *args: clear("actor", *args)
    trainer.ref = SimpleNamespace(
        compute_logp=lambda batch: [0.0 for _ in batch],
        get_device_stats=lambda: _FakeDeviceStats(),
        clear_batches=lambda *args: clear("ref", *args),
    )
    trainer.critic = SimpleNamespace(
        compute_values=lambda batch: [0.0 for _ in batch],
        get_device_stats=lambda: _FakeDeviceStats(),
        ppo_update=lambda batch: events.append(("critic_update", {})),
        step_lr_scheduler=lambda: None,
        clear_batches=lambda *args: clear("critic", *args),
    )
    trainer.data_controller = SimpleNamespace(clear_batches=lambda: clear("data"))
    trainer.rollout = SimpleNamespace(
        pause=lambda: None,
        pause_generation_sync=lambda: None,
        offload=lambda tags: None,
    )
    trainer.weight_update_meta = SimpleNamespace(with_version=lambda version: version)
    trainer._update_weights_and_publish_version = lambda *args: after_release("publish")
    trainer._save_hf = lambda **kwargs: after_release("save_hf")
    trainer._save_recover_checkpoint = lambda **kwargs: after_release("save_recover")
    trainer._evaluate = lambda **kwargs: after_release("eval")
    trainer._save_perf_tracer = lambda **kwargs: None

    if cleanup_fails:
        with pytest.raises(RuntimeError, match="actor cleanup failed"):
            trainer.train(workflow="test-workflow", eval_workflow="test-eval")
    else:
        trainer.train(workflow="test-workflow", eval_workflow="test-eval")

    phases = [name for name, _ in events]
    expected = [
        "actor_update",
        "critic_update",
        "clear_actor",
        "clear_critic",
        "clear_ref",
        "clear_data",
    ]
    if not cleanup_fails:
        expected += (
            ["save_hf", "save_recover", "publish"]
            if mode == "awex"
            else ["publish", "save_hf", "save_recover"]
        ) + ["eval", "commit"]
    assert phases == expected


@pytest.mark.parametrize(
    "version,mode,offload,expected",
    [
        ("v1", "awex", True, True),
        ("v1", "awex", False, False),
        ("v1", "xccl", True, False),
        ("v2", "awex", True, False),
    ],
)
def test_awex_reference_initialization_uses_independent_residency(
    monkeypatch,
    version,
    mode,
    offload,
    expected,
):
    """Only an offloaded v1 AWEX reference needs manual residency before rollout init."""
    _disable_timing_contexts(monkeypatch, rl_trainer)
    events = []
    trainer = _build_ppo_trainer(events)
    trainer.config.actor._version = version
    trainer.config.actor.weight_update_mode = mode
    trainer.config.ref = SimpleNamespace(backend="megatron:d1p1t1")
    trainer._should_offload_ref = offload
    trainer.ref = SimpleNamespace(
        init_weight_residency_adapter=lambda: events.append("residency"),
        offload=lambda: events.append("offload"),
    )
    trainer._initialize_awex_reference_residency()
    assert events == (["residency", "offload"] if expected else [])


@pytest.mark.parametrize(
    ("module", "trainer_cls", "update_method"),
    [
        (sft_trainer, SFTTrainer, "train_lm"),
        (dpo_trainer, DPOTrainer, "train_dpo"),
        (rw_trainer, RWTrainer, "train_rw"),
    ],
    ids=["sft", "dpo", "rw"],
)
@pytest.mark.parametrize("recovered", [False, True], ids=["fresh", "recovered"])
def test_supervised_trainers_order_initial_eval_around_recovery(
    monkeypatch,
    module,
    trainer_cls,
    update_method: str,
    recovered: bool,
):
    """Trainer call sites should run a baseline only before a fresh update."""
    _disable_timing_contexts(monkeypatch, module)
    events: list[tuple] = []
    trainer = _build_supervised_trainer(
        trainer_cls,
        update_method,
        events,
        recovered=recovered,
    )

    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train()

    if recovered:
        assert [event for event, _ in events] == ["update"]
    else:
        assert [event for event, _ in events] == ["initial_eval", "commit", "update"]
        assert events[1][1] == {
            "epoch": -1,
            "epoch_step": -1,
            "global_step": -1,
        }


@pytest.mark.parametrize("recovered", [False, True], ids=["fresh", "recovered"])
def test_ppo_trainer_orders_initial_eval_around_recovery(monkeypatch, recovered: bool):
    """PPO should record a version-zero baseline only on a fresh run."""
    _disable_timing_contexts(monkeypatch, rl_trainer)
    events: list[tuple] = []
    trainer = _build_ppo_trainer(events, recovered=recovered)

    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train(workflow=object())

    if recovered:
        assert [event for event, _ in events] == ["update"]
    else:
        assert [event for event, _ in events] == ["initial_eval", "commit", "update"]
        assert events[1][1] == {
            "epoch": -1,
            "epoch_step": -1,
            "global_step": -1,
        }


def test_ppo_initial_eval_offloads_rollout_when_evaluation_fails(monkeypatch):
    """A failed colocated baseline must still restore the offloaded state."""
    _disable_timing_contexts(monkeypatch, rl_trainer)
    events: list[str] = []
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.eval_rollout = object()
    trainer.valid_dataloader = object()
    trainer.evaluator = _CallbackEvaluator()
    trainer._should_offload_rollout = True
    trainer._onload_rollout = lambda *, is_eval: events.append(f"onload:{is_eval}")
    trainer._offload_rollout = lambda *, is_eval: events.append(f"offload:{is_eval}")

    def fail_evaluation(**_kwargs):
        events.append("evaluate")
        raise RuntimeError("evaluation failed")

    trainer._evaluate_fn = fail_evaluation

    with pytest.raises(RuntimeError, match="evaluation failed"):
        trainer._evaluate_before_train(
            eval_workflow=object(),
            eval_workflow_kwargs=None,
        )

    assert events == ["onload:True", "evaluate", "offload:True"]


def test_ppo_consumes_initial_eval_without_complete_evaluation_inputs():
    """PPO should consume the baseline when no evaluation workflow is provided."""
    evaluator = _RecordingEvaluator()
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.eval_rollout = object()
    trainer.valid_dataloader = object()
    trainer.evaluator = evaluator

    ran = trainer._evaluate_before_train(
        eval_workflow=None,
        eval_workflow_kwargs=None,
    )

    assert ran is False
    assert evaluator.callbacks == [None]


def test_initial_eval_stats_use_step_zero_before_first_update(monkeypatch):
    """Baseline and first-update metrics should occupy distinct log steps."""
    stats_logger = StatsLogger.__new__(StatsLogger)
    stats_logger.ft_spec = SimpleNamespace(
        total_train_epochs=1,
        steps_per_epoch=1,
        total_train_steps=1,
    )
    stats_logger._last_commit_step = -1
    stats_logger._trackio_enabled = False
    stats_logger.summary_writer = None
    stats_logger.print_stats = lambda _stats: None
    logged_steps: list[int] = []

    monkeypatch.setattr(stats_logger_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        stats_logger_module.wandb,
        "log",
        lambda _data, *, step: logged_steps.append(step),
    )
    monkeypatch.setattr(
        stats_logger_module.swanlab,
        "log",
        lambda _data, *, step: None,
    )

    stats_logger.commit(
        epoch=-1,
        step=-1,
        global_step=-1,
        data={"eval/reward": 0.5},
    )
    stats_logger.commit(
        epoch=0,
        step=0,
        global_step=0,
        data={"train/loss": 0.1},
    )

    assert logged_steps == [0, 1]
    assert stats_logger.state_dict() == {"last_commit_step": 1}
