"""Tests for Megatron-Bridge deterministic provider configuration."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch


class _FinalizeReached(RuntimeError):
    pass


class _FakeProvider(SimpleNamespace):
    def finalize(self) -> None:
        self.config_at_finalize = (
            self.deterministic_mode,
            self.attention_backend,
            self.cross_entropy_loss_fusion,
            self.bias_dropout_fusion,
        )
        raise _FinalizeReached


class _FakeBridge:
    def __init__(self, provider: _FakeProvider) -> None:
        self.provider = provider

    def to_megatron_provider(self, *, load_weights: bool) -> _FakeProvider:
        assert load_weights is False
        return self.provider


@pytest.fixture(autouse=True)
def _restore_global_deterministic_state(monkeypatch):
    monkeypatch.delenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    yield

    torch.use_deterministic_algorithms(
        previous_enabled,
        warn_only=previous_warn_only,
    )


def _make_mcore_config(*, deterministic: bool) -> SimpleNamespace:
    return SimpleNamespace(
        virtual_pipeline_parallel_size=None,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        distribute_saved_activations=False,
        recompute_modules=None,
        enable_mtp=False,
        moe_token_dispatcher_type="alltoall",
        use_deterministic_algorithms=deterministic,
    )


def _make_provider(attention_backend) -> _FakeProvider:
    return _FakeProvider(
        deterministic_mode=False,
        attention_backend=attention_backend,
        cross_entropy_loss_fusion=True,
        bias_dropout_fusion=True,
        mtp_num_layers=None,
    )


def _run_until_provider_finalize(
    provider: _FakeProvider,
    *,
    deterministic: bool,
) -> None:
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore.registry import make_mcore_model

    tf_config = SimpleNamespace(params_dtype=None)
    mcore_config = _make_mcore_config(deterministic=deterministic)

    with (
        mock.patch.multiple(
            "areal.models.mcore.registry.mpu",
            get_tensor_model_parallel_world_size=mock.DEFAULT,
            get_pipeline_model_parallel_world_size=mock.DEFAULT,
            get_context_parallel_world_size=mock.DEFAULT,
            get_expert_model_parallel_world_size=mock.DEFAULT,
            get_expert_tensor_parallel_world_size=mock.DEFAULT,
        ) as mpu_mocks,
        pytest.raises(_FinalizeReached),
    ):
        for getter in mpu_mocks.values():
            getter.return_value = 1
        make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=tf_config,
            mcore_config=mcore_config,
            bridge=_FakeBridge(provider),
            bridge_type="megatron-bridge",
        )


def test_megatron_bridge_provider_applies_determinism_before_finalize():
    """Deterministic settings reach the actual provider before finalization."""
    enums = pytest.importorskip("megatron.core.transformer.enums")
    provider = _make_provider(enums.AttnBackend.auto)

    _run_until_provider_finalize(provider, deterministic=True)

    assert provider.config_at_finalize == (
        True,
        enums.AttnBackend.flash,
        False,
        False,
    )


def test_megatron_bridge_provider_preserves_defaults_when_disabled():
    """The Megatron-Bridge provider remains unchanged without the opt-in."""
    attention_backend = object()
    provider = _make_provider(attention_backend)

    _run_until_provider_finalize(provider, deterministic=False)

    assert provider.config_at_finalize == (
        False,
        attention_backend,
        True,
        True,
    )
