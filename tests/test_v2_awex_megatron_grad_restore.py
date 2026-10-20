from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter
from areal.engine.megatron_engine import MegatronEngine


class _StopAfterZeroGrad(Exception):
    pass


def test_train_batch_restores_residency_grad_buffers_before_zero_grad(monkeypatch):
    events = []
    monkeypatch.setattr(
        "areal.engine.megatron_engine.record_training_batch",
        lambda *args: nullcontext(),
    )

    def zero_grad():
        events.append("zero_grad")
        raise _StopAfterZeroGrad

    engine = SimpleNamespace(
        _weight_residency=SimpleNamespace(
            ensure_grad_buffers=lambda: events.append("restore_grad_buffers")
        ),
        _ensure_ready=lambda: events.append("ensure_ready"),
        optimizer_zero_grad=zero_grad,
        _normalize_batch_input=lambda batch: (batch, None),
        hf_config=SimpleNamespace(),
    )
    with pytest.raises(_StopAfterZeroGrad):
        MegatronEngine.train_batch(
            engine,
            input_={},
            loss_fn=lambda *_args: None,
            loss_weight_fn=lambda *_args: None,
        )
    assert events == ["ensure_ready", "restore_grad_buffers", "zero_grad"]


def test_colocate_enable_shares_engine_residency_and_teardown_preserves_it():
    state = SimpleNamespace(released_tags=frozenset({"weights", "optimizer"}))
    engine = SimpleNamespace(
        _awex_adapter=None, _ensure_weight_residency=Mock(return_value=state)
    )
    adapter = AwexMegatronAdapter(engine)
    assert engine._awex_adapter is None

    adapter.enable_colocate_memory_management()
    adapter.teardown_colocate_weight_update()

    assert engine._awex_adapter is adapter
    assert adapter.residency is state
    assert adapter._released_tags == {"weights", "optimizer"}


def test_failed_publish_restores_initial_offloaded_weight_state():
    state = SimpleNamespace(is_released=Mock(side_effect=[True, False]))
    adapter = AwexMegatronAdapter(SimpleNamespace())
    adapter._residency = state
    adapter._colocate_init_fingerprint = ("pair",)
    adapter._execute_meta_colocate_weight_update = Mock(
        side_effect=RuntimeError("publish")
    )
    adapter.release_memory = Mock()

    with pytest.raises(RuntimeError, match="publish"):
        adapter.execute_colocate_weight_update(1)

    adapter.release_memory.assert_called_once_with(tags=["weights"])
