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
    handler._warmup_communicators = lambda _engines: events.append("warmup")
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
    """A GatewayTrainController stand-in whose train_phase mirrors the real one."""
    import contextlib

    @contextlib.contextmanager
    def train_phase():
        events.append("enter_train_phase")
        rollout.pause()
        rollout.pause_generation_sync()
        rollout.offload(tags=["kv_cache"])
        rollout.offload(tags=["cuda_graph"])
        rollout.offload(tags=["weights"])
        events.append(("awex", "resume_memory", ("weights", "optimizer")))
        try:
            yield
        finally:
            events.append("exit_train_phase")
            events.append(("awex", "release_memory", ("weights", "optimizer")))
            rollout.onload(tags=["cuda_graph"])
            rollout.onload(tags=["kv_cache"])
            rollout.continue_generation()
            rollout.resume()

    actor = SimpleNamespace(
        _colocate=colocate,
        train_phase=train_phase,
        connect_engine=lambda r, m: events.append("connect_engine"),
        update_weights=lambda meta: events.append(("update_weights", meta.version)),
        set_version=lambda v: events.append(("actor_set_version", v)),
    )
    return actor


def _run_load(events, actor, rollout, colocate=True):
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

        assert "enter_train_phase" in events, (
            "recover must reuse train_phase instead of hand-rolling the handover"
        )
        assert (
            events.index("enter_train_phase")
            < events.index(("update_weights", 8))
            < events.index("exit_train_phase")
        )

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
