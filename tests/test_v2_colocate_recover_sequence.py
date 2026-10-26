"""Tests for the colocate recover weight-restore sequence."""

from types import SimpleNamespace
from unittest import mock

import pytest

from areal.utils.recover import RecoverHandler


@pytest.fixture(autouse=True)
def published_checkpoint_source(monkeypatch):
    source = SimpleNamespace(
        manifest="generation/recover_info",
        payloads={"default": "generation/payloads/default"},
        label="generation_step00000007",
        transactional=True,
    )
    monkeypatch.setattr(
        "areal.utils.recover.checkpoint_pointer.resolve_checkpoint",
        lambda *args: source,
    )


def _make_handler(events):
    handler = object.__new__(RecoverHandler)
    handler.config = SimpleNamespace(
        mode="auto",
        experiment_name="exp",
        trial_name="trial",
        fileroot="/tmp/does-not-matter",
    )
    handler.freq_ctl = SimpleNamespace(load_state_dict=lambda _d: None)
    handler._load_checkpoint = lambda engine_, path, name="default": events.append(
        ("load_checkpoint", name)
    )
    return handler


def _make_rollout(events):
    def rec(name):
        def _fn(*args, **kwargs):
            tags = kwargs.get("tags", args[0] if args else None)
            events.append((name, tuple(tags)) if tags is not None else name)

        return _fn

    return SimpleNamespace(
        pause=rec("pause"),
        resume=rec("resume"),
        pause_generation_sync=rec("pause_generation_sync"),
        continue_generation=rec("continue_generation"),
        offload=rec("offload"),
        onload=rec("onload"),
        set_version=lambda v: events.append(("rollout_set_version", v)),
    )


def _make_actor(events, rollout, colocate=True):
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )

    actor = object.__new__(GatewayTrainController)
    actor._colocate = colocate
    actor.rollout = rollout
    actor._broadcast_awex_memory_op = lambda op, tags: events.append(
        ("awex", op, tuple(tags))
    )
    actor.connect_engine = lambda r, m: events.append("connect_engine")

    def update_weights(meta):
        actor._colocate_update_started = True
        actor._colocate_transfer_succeeded = True
        events.append(("update_weights", meta.version))
        actor._colocate_update_succeeded = False

    actor.update_weights = update_weights
    actor.set_version = lambda v: events.append(("actor_set_version", v))

    def mark_committed():
        events.append("mark_committed")
        actor._colocate_update_succeeded = True

    actor.mark_colocate_update_committed = mark_committed
    return actor


def _run_load(events, actor, rollout, colocate=True, connected=False):
    meta = SimpleNamespace(
        type="awex",
        colocate=colocate,
        with_version=lambda v: SimpleNamespace(version=v, type="awex"),
    )
    recover_info = SimpleNamespace(
        last_step_info=SimpleNamespace(global_step=7, next=lambda: "step-8"),
        saver_info={},
        evaluator_info={},
        stats_logger_info={},
        dataloader_info={},
        checkpoint_info={},
    )
    noop = SimpleNamespace(load_state_dict=lambda _d: None)
    with mock.patch("areal.utils.recover.RecoverInfo.load", return_value=recover_info):
        return RecoverHandler.load(
            _make_handler(events),
            engine=actor,
            saver=noop,
            evaluator=noop,
            stats_logger=noop,
            dataloader=noop,
            inference_engine=rollout,
            weight_update_meta=meta,
            inference_engine_connected=connected,
        )


@pytest.mark.parametrize("connected", [False, True])
def test_recovery_restores_weights_and_versions_before_resuming(connected):
    events = []
    rollout = _make_rollout(events)
    _run_load(events, _make_actor(events, rollout), rollout, connected=connected)

    expected = [
        ("offload", ("weights",)),
        ("load_checkpoint", "default"),
        ("update_weights", 8),
        ("actor_set_version", 8),
        ("rollout_set_version", 8),
        "mark_committed",
        ("onload", ("kv_cache",)),
        "continue_generation",
    ]
    positions = [events.index(event) for event in expected]
    assert positions == sorted(positions)
    assert ("connect_engine" in events) is not connected


def test_generation_stays_paused_when_checkpoint_load_fails():
    events = []
    rollout = _make_rollout(events)
    actor = _make_actor(events, rollout)
    handler = _make_handler(events)
    handler._load_checkpoint = mock.Mock(side_effect=RuntimeError("checkpoint corrupt"))
    with mock.patch(__name__ + "._make_handler", return_value=handler):
        with pytest.raises(RuntimeError, match="corrupt"):
            _run_load(events, actor, rollout)

    assert "continue_generation" not in events
    assert "resume" not in events
