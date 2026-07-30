from types import SimpleNamespace

import pytest

from areal.engine.megatron_engine import MegatronEngine
from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter


class _StopAfterZeroGrad(Exception):
    pass


def test_train_batch_restores_awex_grad_buffers_before_zero_grad():
    events = []

    def optimizer_zero_grad():
        events.append("zero_grad")
        raise _StopAfterZeroGrad

    engine = SimpleNamespace(
        _awex_adapter=SimpleNamespace(
            ensure_grad_buffers=lambda: events.append("restore_grad_buffers")
        ),
        _ensure_ready=lambda: events.append("ensure_ready"),
        optimizer_zero_grad=optimizer_zero_grad,
    )

    with pytest.raises(_StopAfterZeroGrad):
        MegatronEngine.train_batch(
            engine,
            input_={},
            loss_fn=lambda *_args, **_kwargs: None,
            loss_weight_fn=lambda *_args, **_kwargs: None,
        )

    assert events == ["ensure_ready", "restore_grad_buffers", "zero_grad"]


def test_awex_megatron_adapter_registers_with_its_engine():
    engine = SimpleNamespace()

    adapter = AwexMegatronAdapter(engine)

    assert engine._awex_adapter is adapter
