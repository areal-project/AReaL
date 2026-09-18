"""Unit tests for the SGLang fork contract checks (no GPU needed)."""

import pytest
import torch

from areal.engine.sglang_fork_contract import (
    check_scheduler_contract,
    check_static_contract,
)


class _RunningBatch:
    def is_empty(self):
        return True


class _ModelRunner:
    model = object()


class _TpWorker:
    tp_rank = 3
    model_runner = _ModelRunner()


class _GoodScheduler:
    _engine_paused = False
    tp_worker = _TpWorker()
    running_batch = _RunningBatch()
    waiting_queue = []
    weight_updater = object()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang kernel imports require CUDA"
)
def test_static_contract_accepts_pinned_sglang_runtime():
    """The supported 0.5.10 legacy idle gate passes startup checks."""
    check_static_contract()


def test_good_scheduler_passes():
    check_scheduler_contract(_GoodScheduler())


def test_scheduler_level_tp_rank_also_accepted():
    class S(_GoodScheduler):
        tp_rank = 5

    check_scheduler_contract(S())


def test_direct_scheduler_memory_api_does_not_require_weight_updater():
    class S(_GoodScheduler):
        weight_updater = None

        def release_memory_occupation(self, request):
            return request

        def resume_memory_occupation(self, request):
            return request

    check_scheduler_contract(S())


def test_missing_tp_rank_fails(monkeypatch):
    class NoRankWorker:
        model_runner = _ModelRunner()

    class S(_GoodScheduler):
        tp_worker = NoRankWorker()

    with pytest.raises(RuntimeError, match="tp_rank"):
        check_scheduler_contract(S())


def test_missing_engine_paused_fails():
    class S:
        tp_worker = _TpWorker()
        running_batch = _RunningBatch()
        waiting_queue = []
        weight_updater = object()

    with pytest.raises(RuntimeError, match="_engine_paused"):
        check_scheduler_contract(S())


def test_all_violations_reported_together():
    class Bare:
        pass

    with pytest.raises(RuntimeError) as exc:
        check_scheduler_contract(Bare())
    message = str(exc.value)
    for needle in ("_engine_paused", "tp_rank", "weight_updater"):
        assert needle in message


def test_warn_mode_does_not_raise(monkeypatch):
    class Bare:
        pass

    monkeypatch.setenv("AREAL_SGLANG_CONTRACT", "warn")
    check_scheduler_contract(Bare())


def test_off_mode_skips(monkeypatch):
    class Bare:
        pass

    monkeypatch.setenv("AREAL_SGLANG_CONTRACT", "off")
    check_scheduler_contract(Bare())


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("location", ["scheduler", "worker", "runner"])
def test_native_parallel_state_preserves_each_tensor_parallel_rank(rank, location):
    from types import SimpleNamespace

    from areal.engine.sglang_fork_contract import resolve_scheduler_parallel_attr

    runner = SimpleNamespace(model=object())
    worker = SimpleNamespace(model_runner=runner)
    scheduler = SimpleNamespace(
        _engine_paused=False,
        tp_worker=worker,
        running_batch=_RunningBatch(),
        waiting_queue=[],
        weight_updater=object(),
    )
    owner = {"scheduler": scheduler, "worker": worker, "runner": runner}[location]
    owner.ps = SimpleNamespace(tp_rank=rank, tp_size=4, pp_rank=0)
    check_scheduler_contract(scheduler)
    assert resolve_scheduler_parallel_attr(scheduler, "tp_rank") == rank
    assert resolve_scheduler_parallel_attr(scheduler, "pp_rank") == 0
    assert resolve_scheduler_parallel_attr(scheduler, "attn_tp_rank") is None
