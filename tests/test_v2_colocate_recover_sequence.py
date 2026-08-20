"""Tests for the colocate recover weight-restore sequence."""

from types import SimpleNamespace
from unittest import mock

import pytest

from areal.utils.recover import RecoverHandler


def _make_handler(events):
    handler = object.__new__(RecoverHandler)
    handler.config = SimpleNamespace(
        mode="auto",
        experiment_name="exp",
        trial_name="trial",
        fileroot="/tmp/does-not-matter",
    )
    handler.freq_ctl = SimpleNamespace(load_state_dict=lambda _d: None)
    handler._load_checkpoint = lambda engine_, name="default": events.append(
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


class TestColocateRecoverSequence:
    def test_kv_cache_is_onloaded_before_generation_resumes(self):
        events = []
        rollout = _make_rollout(events)
        _run_load(events, _make_actor(events, rollout), rollout)

        assert ("onload", ("kv_cache",)) in events, (
            "recover released kv_cache but never mapped it back; prefill would "
            "run against unmapped pages"
        )
        assert "continue_generation" in events, (
            "pause_generation_sync closed the generation gate and nothing reopened it"
        )
        assert events.index(("onload", ("kv_cache",))) < events.index(
            "continue_generation"
        )

    def test_checkpoint_loads_while_rollout_memory_is_released(self):
        events = []
        rollout = _make_rollout(events)
        _run_load(events, _make_actor(events, rollout), rollout)

        load_idx = events.index(("load_checkpoint", "default"))
        assert events.index(("offload", ("weights",))) < load_idx
        assert load_idx < events.index(("update_weights", 8))

    def test_weight_update_runs_inside_the_train_phase(self):
        events = []
        rollout = _make_rollout(events)
        _run_load(events, _make_actor(events, rollout), rollout)

        assert events.index(("offload", ("weights",))) < events.index(
            ("update_weights", 8)
        )
        assert events.index(("update_weights", 8)) < events.index(
            ("onload", ("kv_cache",))
        )

    def test_preconnected_trainer_does_not_connect_a_second_gateway(self):
        events = []
        rollout = _make_rollout(events)
        _run_load(events, _make_actor(events, rollout), rollout, connected=True)

        assert "connect_engine" not in events

    def test_versions_commit_before_generation_resumes(self):
        events = []
        rollout = _make_rollout(events)
        _run_load(events, _make_actor(events, rollout), rollout)

        assert events.index(("actor_set_version", 8)) < events.index("mark_committed")
        assert events.index(("rollout_set_version", 8)) < events.index("mark_committed")
        assert events.index("mark_committed") < events.index("continue_generation")

    def test_generation_resumes_even_when_checkpoint_load_fails(self):
        events = []
        rollout = _make_rollout(events)
        handler_events = events
        actor = _make_actor(events, rollout)

        def boom(engine_, name="default"):
            handler_events.append(("load_checkpoint", name))
            raise RuntimeError("checkpoint is corrupt")

        with pytest.raises(RuntimeError, match="corrupt"):
            meta = SimpleNamespace(
                type="awex",
                colocate=True,
                with_version=lambda v: SimpleNamespace(version=v, type="awex"),
            )
            recover_info = SimpleNamespace(
                last_step_info=SimpleNamespace(global_step=7, next=lambda: "s"),
                saver_info={},
                evaluator_info={},
                stats_logger_info={},
                dataloader_info={},
                checkpoint_info={},
            )
            noop = SimpleNamespace(load_state_dict=lambda _d: None)
            handler = _make_handler(events)
            handler._load_checkpoint = boom
            with mock.patch(
                "areal.utils.recover.RecoverInfo.load", return_value=recover_info
            ):
                RecoverHandler.load(
                    handler,
                    engine=actor,
                    saver=noop,
                    evaluator=noop,
                    stats_logger=noop,
                    dataloader=noop,
                    inference_engine=rollout,
                    weight_update_meta=meta,
                )

        assert "continue_generation" in events
        assert "resume" in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
