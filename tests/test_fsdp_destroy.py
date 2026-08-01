from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from areal.engine.fsdp_engine import FSDPEngine

MODULE = "areal.engine.fsdp_engine"


def _make_engine(*, is_offload: bool) -> FSDPEngine:
    engine = object.__new__(FSDPEngine)
    engine._initialized = True
    engine._destroyed = False
    engine.own_global_group = True
    engine._cpu_group = object()
    engine.is_offload = is_offload
    engine._offload_depth = 0
    engine._per_layer_optim_wrapper = None
    engine.optimizer = MagicMock()
    engine.model = MagicMock()
    engine.logger = MagicMock()
    engine.get_device_stats = MagicMock(return_value=MagicMock())
    return engine


def test_destroy_offloaded_engine_resumes_before_releasing_resources():
    engine = _make_engine(is_offload=True)
    events: list[str] = []
    tms = MagicMock()

    def _resume_while_resources_are_live() -> None:
        assert hasattr(engine, "optimizer")
        assert hasattr(engine, "model")
        events.append("resume")

    tms.resume.side_effect = _resume_while_resources_are_live
    platform = MagicMock()
    platform.synchronize.side_effect = lambda: events.append("synchronize")
    platform.empty_cache.side_effect = lambda: events.append("empty_cache")

    with (
        patch(f"{MODULE}.is_tms_enabled", return_value=False),
        patch(f"{MODULE}.torch_memory_saver", tms),
        patch(f"{MODULE}.current_platform", platform),
        patch(f"{MODULE}.gc.collect"),
        patch(f"{MODULE}.dist.is_initialized", return_value=True),
        patch(
            f"{MODULE}.dist.monitored_barrier",
            side_effect=lambda *, group, **_kwargs: events.append("barrier"),
        ) as barrier,
        patch(
            f"{MODULE}.dist.destroy_process_group",
            side_effect=lambda: events.append("destroy_process_group"),
        ),
    ):
        engine.destroy()

    assert events == [
        "resume",
        "synchronize",
        "empty_cache",
        "barrier",
        "destroy_process_group",
    ]
    barrier.assert_called_once_with(
        group=engine._cpu_group,
        timeout=timedelta(seconds=10),
        wait_all_ranks=True,
    )
    assert engine.is_offload is False
    assert engine._initialized is False
    assert engine._destroyed is True
    assert engine.own_global_group is False


def test_destroy_active_engine_is_idempotent():
    engine = _make_engine(is_offload=False)
    tms = MagicMock()
    platform = MagicMock()

    with (
        patch(f"{MODULE}.is_tms_enabled", return_value=True),
        patch(f"{MODULE}.torch_memory_saver", tms),
        patch(f"{MODULE}.current_platform", platform),
        patch(f"{MODULE}.gc.collect"),
        patch(f"{MODULE}.dist.is_initialized", return_value=True),
        patch(f"{MODULE}.dist.monitored_barrier") as barrier,
        patch(f"{MODULE}.dist.destroy_process_group") as destroy_group,
    ):
        engine.destroy()
        engine.destroy()

    tms.resume.assert_not_called()
    platform.empty_cache.assert_called_once_with()
    barrier.assert_called_once_with(
        group=engine._cpu_group,
        timeout=timedelta(seconds=10),
        wait_all_ranks=True,
    )
    destroy_group.assert_called_once_with()


def test_destroy_continues_after_monitored_barrier_failure():
    engine = _make_engine(is_offload=False)

    with (
        patch(f"{MODULE}.is_tms_enabled", return_value=True),
        patch(f"{MODULE}.current_platform") as platform,
        patch(f"{MODULE}.gc.collect"),
        patch(f"{MODULE}.dist.is_initialized", return_value=True),
        patch(
            f"{MODULE}.dist.monitored_barrier",
            side_effect=RuntimeError("rank 1 missing"),
        ),
        patch(f"{MODULE}.dist.destroy_process_group") as destroy_group,
    ):
        engine.destroy()

    platform.empty_cache.assert_called_once_with()
    destroy_group.assert_called_once_with()
    engine.logger.warning.assert_called_once()
    assert engine._destroyed is True
    assert engine.own_global_group is False


def test_initialize_starts_new_destroy_lifecycle():
    engine = _make_engine(is_offload=False)
    engine._destroyed = True

    with (
        patch(f"{MODULE}.pkg_version.is_version_less", return_value=False),
        patch(f"{MODULE}.is_tms_enabled", return_value=False),
        patch.object(
            engine,
            "_create_device_model",
            side_effect=RuntimeError("stop after lifecycle reset"),
        ),
        pytest.raises(RuntimeError, match="stop after lifecycle reset"),
    ):
        engine.initialize(addr=None, ft_spec=MagicMock())

    assert engine._destroyed is False


def test_offload_commits_state_before_barrier_failure():
    engine = _make_engine(is_offload=False)
    tms = MagicMock()
    platform = MagicMock()

    with (
        patch(f"{MODULE}.is_tms_enabled", return_value=True),
        patch(f"{MODULE}.torch_memory_saver", tms),
        patch(f"{MODULE}.current_platform", platform),
        patch(f"{MODULE}.dist.barrier", side_effect=RuntimeError("barrier failed")),
        pytest.raises(RuntimeError, match="barrier failed"),
    ):
        engine.offload()

    tms.pause.assert_called_once_with()
    assert engine.is_offload is True


def test_onload_commits_state_before_barrier_failure():
    engine = _make_engine(is_offload=True)
    tms = MagicMock()
    platform = MagicMock()

    with (
        patch(f"{MODULE}.torch_memory_saver", tms),
        patch(f"{MODULE}.current_platform", platform),
        patch(f"{MODULE}.dist.barrier", side_effect=RuntimeError("barrier failed")),
        pytest.raises(RuntimeError, match="barrier failed"),
    ):
        engine.onload()

    tms.resume.assert_called_once_with()
    assert engine.is_offload is False


def test_destroy_resume_failure_leaves_engine_retryable():
    engine = _make_engine(is_offload=True)
    tms = MagicMock()
    tms.resume.side_effect = RuntimeError("resume failed")
    platform = MagicMock()

    with (
        patch(f"{MODULE}.is_tms_enabled", return_value=True),
        patch(f"{MODULE}.torch_memory_saver", tms),
        patch(f"{MODULE}.current_platform", platform),
        pytest.raises(RuntimeError, match="resume failed"),
    ):
        engine.destroy()

    platform.synchronize.assert_not_called()
    platform.empty_cache.assert_not_called()
    assert engine.is_offload is True
    assert engine._initialized is True
    assert engine._destroyed is False
