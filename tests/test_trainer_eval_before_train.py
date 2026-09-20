# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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


class _FakeSaver:
    def maybe_wait_for_staging(self) -> None:
        pass


class _FakeDataLoader(list):
    sampler = None


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
    parallel_strategy = SimpleNamespace(dp_size=1)

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
    trainer.train_dataloader = _FakeDataLoader([[{}], [{}]] if recovered else [[{}]])
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
        rollout=SimpleNamespace(agent=None, _version="v1"),
        teacher=None,
        gconfig=SimpleNamespace(
            n_samples=1,
            reward_normalization=False,
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
    trainer.recover_info = (
        SimpleNamespace(last_step_info=_FakeLastStepInfo()) if recovered else None
    )
    trainer.train_dataloader = [[{}], [{}]] if recovered else [[{}]]
    trainer.saver = _FakeSaver()
    trainer.actor = _FailingPPOActor(events)
    trainer.critic = None
    trainer.ref = None
    trainer.teacher = None
    trainer.mopd_execution_plan = None
    trainer.mopd_teacher_phase = None
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._requires_proxy_workflow = lambda _workflow: False
    trainer._evaluate_before_train = lambda *_args, **_kwargs: _record_initial_eval(
        events
    )
    trainer._export_and_commit_stats = lambda **kwargs: _record_commit(events, **kwargs)
    return trainer


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


@pytest.mark.parametrize("role", ["ref", "critic", "teacher", None])
@pytest.mark.parametrize(
    ("version", "mode"), [("v1", "awex"), ("v1", "disk"), ("v2", "awex")]
)
def test_ppo_train_colocated_auxiliary_scoring_releases_rollout_first(
    monkeypatch, role, version, mode
):
    """Auxiliary scoring must not overlap rollout memory or load the actor early."""
    _disable_timing_contexts(monkeypatch, rl_trainer)
    trainer = _build_ppo_trainer([])
    trainer.config.actor._version = version
    trainer.config.actor.weight_update_mode = mode
    trainer.config.rollout._version = version
    awex_colocate = version == "v1" and mode == "awex"
    trainer._should_offload_rollout = not awex_colocate
    trainer._should_offload_actor = not awex_colocate
    trainer.rollout = Mock()
    released = set()
    memory_tags = {"kv_cache", "weights", "cuda_graph"}

    def release_rollout(*, tags=None):
        trainer.rollout.pause.assert_called_once()
        pause = (
            trainer.rollout.pause_generation_sync
            if awex_colocate
            else trainer.rollout.pause_generation
        )
        pause.assert_called_once()
        released.update(memory_tags if tags is None else tags)

    trainer.rollout.offload.side_effect = release_rollout
    trainer.actor.onload = Mock()
    auxiliary = Mock(parallel_strategy=SimpleNamespace(dp_size=1))

    def check_auxiliary_memory():
        assert released == memory_tags, "rollout memory is still resident"
        trainer.actor.onload.assert_not_called()

    def score(batch):
        check_auxiliary_memory()
        return [object() for _ in batch]

    auxiliary.onload.side_effect = check_auxiliary_memory
    auxiliary.compute_logp.side_effect = score
    auxiliary.compute_values.side_effect = score
    if role is not None:
        setattr(trainer, role, auxiliary)
        setattr(trainer, f"_should_offload_{role}", True)
    if role == "teacher":
        trainer.config.teacher = SimpleNamespace(
            engine_type="train", rl_loss_weight=1.0, distill_loss_weight=1.0
        )

    def check_actor_memory():
        assert released == memory_tags
        if role is not None:
            scorer = (
                auxiliary.compute_values if role == "critic" else auxiliary.compute_logp
            )
            scorer.assert_called_once()
            if role != "critic":
                auxiliary.offload.assert_called_once()

    trainer.actor.onload.side_effect = check_actor_memory

    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train(workflow=object())

    trainer.actor.onload.assert_called_once()
    assert trainer.rollout.offload.call_count == (3 if awex_colocate else 1)


@pytest.mark.parametrize(
    ("requires_rl", "explicit_minimum", "expected"),
    [(False, None, 1), (False, 3, 3), (True, None, 2)],
)
def test_rollout_minimum_uses_only_active_rl_estimator(
    monkeypatch, requires_rl, explicit_minimum, expected
):
    from areal.api.cli_args import NormConfig, PPOActorConfig

    _disable_timing_contexts(monkeypatch, rl_trainer)
    trainer = _build_ppo_trainer([])
    trainer.config.actor = PPOActorConfig(
        reward_norm=NormConfig(mean_level="group", std_level="group", group_size=4),
        min_usable_group_size=explicit_minimum,
    )
    trainer.config.gconfig.n_samples = 4
    trainer.mopd_execution_plan = SimpleNamespace(requires_rl=requires_rl)

    def prepare_batch(*args, **kwargs):
        assert kwargs["min_usable_group_size"] == expected
        raise _StopAfterFirstUpdate

    trainer.actor.prepare_batch = prepare_batch
    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train(workflow=object())


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
