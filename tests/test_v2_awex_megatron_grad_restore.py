# SPDX-License-Identifier: Apache-2.0

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


def test_awex_megatron_adapter_registers_only_when_colocation_is_enabled():
    engine = SimpleNamespace(_awex_adapter=None)

    adapter = AwexMegatronAdapter(engine)

    assert engine._awex_adapter is None

    adapter.enable_colocate_memory_management()

    assert engine._awex_adapter is adapter


def test_colocate_teardown_clears_hook_with_released_memory():
    engine = SimpleNamespace(_awex_adapter=None)
    adapter = AwexMegatronAdapter(engine)
    adapter.enable_colocate_memory_management()
    adapter._released_tags.add("weights")

    adapter.teardown_colocate_weight_update()

    assert engine._awex_adapter is None
    assert not adapter._released_tags


def test_adapter_rejects_mixing_separation_and_colocate_modes():
    adapter = AwexMegatronAdapter(SimpleNamespace(_awex_adapter=None))
    adapter._active_mode = "separation"
    adapter._init_fingerprint = ("separation",)

    assert adapter._is_init_retry("separation", ("separation",))

    with pytest.raises(RuntimeError, match="already initialized for separation"):
        adapter._is_init_retry("colocate", ("colocate",))
