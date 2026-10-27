"""Tests for Megatron-Bridge deterministic provider configuration."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from areal.api.cli_args import MegatronEngineConfig


class _FinalizeReached(RuntimeError):
    pass


class _FakeProvider(SimpleNamespace):
    def finalize(self) -> None:
        self.vision_flags_at_finalize = {
            name: getattr(self, name, None)
            for name in ("freeze_vision_model", "freeze_vision_projection")
        }
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


def _make_mcore_config(*, deterministic: bool) -> MegatronEngineConfig:
    return MegatronEngineConfig(
        cross_entropy_loss_fusion=True,
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
    freeze_vision_model: bool | None = None,
    freeze_vision_projection: bool | None = None,
    model_type: str | None = None,
) -> None:
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore.registry import make_mcore_model

    tf_config = SimpleNamespace(params_dtype=None)
    mcore_config = _make_mcore_config(deterministic=deterministic)
    mcore_config.freeze_vision_model = freeze_vision_model
    mcore_config.freeze_vision_projection = freeze_vision_projection

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
            hf_config=SimpleNamespace(model_type=model_type),
            tf_config=tf_config,
            mcore_config=mcore_config,
            bridge=_FakeBridge(provider),
            bridge_type="megatron-bridge",
        )


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("freeze_projection", [False, True])
def test_megatron_bridge_vision_freezing_reaches_provider_before_finalize(
    freeze_encoder, freeze_projection
):
    provider = _make_provider(attention_backend=object())
    provider.freeze_vision_model = False
    provider.freeze_vision_projection = False

    _run_until_provider_finalize(
        provider,
        deterministic=False,
        freeze_vision_model=freeze_encoder,
        freeze_vision_projection=freeze_projection,
    )

    assert provider.vision_flags_at_finalize == {
        "freeze_vision_model": freeze_encoder,
        "freeze_vision_projection": freeze_projection,
    }


def test_megatron_bridge_default_does_not_unfreeze_provider():
    provider = _make_provider(attention_backend=object())
    provider.freeze_vision_model = True
    provider.freeze_vision_projection = True

    _run_until_provider_finalize(provider, deterministic=False)

    assert all(provider.vision_flags_at_finalize.values())


@pytest.mark.parametrize(
    "model_type", ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
)
def test_megatron_bridge_qwen_vision_is_frozen_by_default(model_type):
    provider = _make_provider(attention_backend=object())
    provider.freeze_vision_model = False
    provider.freeze_vision_projection = False
    _run_until_provider_finalize(provider, deterministic=False, model_type=model_type)
    assert all(provider.vision_flags_at_finalize.values())


def test_megatron_bridge_qwen_vision_can_be_explicitly_unfrozen():
    provider = _make_provider(attention_backend=object())
    provider.freeze_vision_model = True
    provider.freeze_vision_projection = True
    _run_until_provider_finalize(
        provider,
        deterministic=False,
        model_type="qwen3_5",
        freeze_vision_model=False,
        freeze_vision_projection=False,
    )
    assert not any(provider.vision_flags_at_finalize.values())


@pytest.mark.parametrize("option", ["freeze_vision_model", "freeze_vision_projection"])
def test_megatron_bridge_rejects_unsupported_vision_freezing(option):
    provider = _make_provider(attention_backend=object())

    with pytest.raises(ValueError, match=f"does not support {option}"):
        _run_until_provider_finalize(provider, deterministic=False, **{option: True})


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


def test_megatron_bridge_rejects_multilayer_mtp_training():
    """AReaL currently supports training one MTP prediction layer."""
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore.registry import make_mcore_model

    provider = _make_provider(attention_backend=object())
    provider.mtp_num_layers = 2
    mcore_config = _make_mcore_config(deterministic=False)
    mcore_config.enable_mtp = True
    mcore_config.enable_mtp_training = True
    mcore_config.mtp_loss_scaling_factor = 0.1

    with (
        mock.patch.multiple(
            "areal.models.mcore.registry.mpu",
            get_tensor_model_parallel_world_size=mock.DEFAULT,
            get_pipeline_model_parallel_world_size=mock.DEFAULT,
            get_context_parallel_world_size=mock.DEFAULT,
            get_expert_model_parallel_world_size=mock.DEFAULT,
            get_expert_tensor_parallel_world_size=mock.DEFAULT,
        ) as mpu_mocks,
        pytest.raises(ValueError, match="supports exactly one prediction layer"),
    ):
        for getter in mpu_mocks.values():
            getter.return_value = 1
        make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=SimpleNamespace(params_dtype=None),
            mcore_config=mcore_config,
            bridge=_FakeBridge(provider),
            bridge_type="megatron-bridge",
        )


def test_megatron_bridge_enables_native_mtp_gradient_isolation():
    """MCore's native detach path must be configured before provider finalize."""
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore.registry import make_mcore_model

    provider = _make_provider(attention_backend=object())
    provider.mtp_num_layers = 1
    provider.mtp_detach_heads = False
    mcore_config = _make_mcore_config(deterministic=False)
    mcore_config.enable_mtp = True
    mcore_config.enable_mtp_training = True
    mcore_config.mtp_loss_scaling_factor = 0.1

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
            tf_config=SimpleNamespace(params_dtype=None),
            mcore_config=mcore_config,
            bridge=_FakeBridge(provider),
            bridge_type="megatron-bridge",
        )

    assert provider.mtp_detach_heads is True
