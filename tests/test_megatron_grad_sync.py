# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.engine import megatron_engine as engine_module


def make_engine(monkeypatch):
    engine = engine_module.MegatronEngine.__new__(engine_module.MegatronEngine)
    engine._has_finalized_model_grads = False
    engine.get_device_stats = Mock(return_value=SimpleNamespace(log=Mock()))
    reclaim = Mock()
    monkeypatch.setattr(engine_module.current_platform, "empty_cache", reclaim)
    return engine, reclaim


def test_first_grad_sync_reclaims_before_reduction_and_preserves_arguments(monkeypatch):
    engine, reclaim = make_engine(monkeypatch)
    gradients = torch.tensor([1.0, -2.0, 3.0])
    model = [SimpleNamespace(gradients=gradients)]
    num_tokens = torch.tensor(4)

    def finalize(models, token_count, *, force_all_reduce=False):
        reclaim.assert_called_once_with()
        assert models is model
        assert token_count is num_tokens
        assert force_all_reduce
        torch.testing.assert_close(gradients, torch.tensor([1.0, -2.0, 3.0]))
        gradients.div_(token_count)

    monkeypatch.setattr(engine_module, "finalize_model_grads", finalize)
    engine._finalize_model_grads(model, num_tokens, force_all_reduce=True)

    torch.testing.assert_close(gradients, torch.tensor([0.25, -0.5, 0.75]))
    assert engine._has_finalized_model_grads


def test_later_grad_sync_keeps_cache_and_still_reduces(monkeypatch):
    engine, reclaim = make_engine(monkeypatch)
    finalize = Mock()
    monkeypatch.setattr(engine_module, "finalize_model_grads", finalize)
    first_model, second_model = object(), object()

    engine._finalize_model_grads(first_model)
    engine._finalize_model_grads(second_model)

    reclaim.assert_called_once_with()
    assert finalize.call_count == 2
    finalize.assert_called_with(second_model)


def test_failed_grad_sync_propagates_without_marking_it_initialized(monkeypatch):
    engine, reclaim = make_engine(monkeypatch)
    failure = RuntimeError("collective failed")
    finalize = Mock(side_effect=[failure, None])
    monkeypatch.setattr(engine_module, "finalize_model_grads", finalize)

    with pytest.raises(RuntimeError, match="collective failed") as caught:
        engine._finalize_model_grads([])

    assert caught.value is failure
    assert not engine._has_finalized_model_grads
    engine._finalize_model_grads([])
    assert reclaim.call_count == 2
    assert engine._has_finalized_model_grads
