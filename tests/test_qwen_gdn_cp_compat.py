# SPDX-License-Identifier: Apache-2.0

import sys
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import pytest

from examples.swe.qwen38_flash_next import gdn_cp_compat


class _BaseConfig:
    def __post_init__(self):
        assert self.context_parallel_size > 0, "positive CP size required"


class _LegacyConfig(_BaseConfig):
    def __post_init__(self):
        super().__post_init__()
        assert self.num_heads % self.context_parallel_size == 0, "head divisibility"
        assert self.context_parallel_size == 1, (
            "Gated delta net does not support context parallel for now"
        )


class _NativeConfig(_BaseConfig):
    def __post_init__(self):
        super().__post_init__()
        assert self.num_heads % self.context_parallel_size == 0, "head divisibility"


class _ChangedGuardConfig:
    def __post_init__(self):
        assert self.context_parallel_size <= 1, (
            "Gated delta net does not support context parallel for now"
        )


class _DuplicateGuardConfig:
    def __post_init__(self):
        assert self.context_parallel_size == 1, (
            "Gated delta net does not support context parallel for now"
        )
        assert self.context_parallel_size == 1, (
            "Gated delta net does not support context parallel for now"
        )


def _legacy_forward(self, cu_seqlens, *, cp_size=2):
    return cu_seqlens // self.cp_size


def _module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _bridge(monkeypatch):
    bridge_gdn = type("GatedDeltaNet", (), {"forward": _legacy_forward})
    for name in ("mcore_bridge", "mcore_bridge.model", "mcore_bridge.model.modules"):
        _module(monkeypatch, name)
    _module(
        monkeypatch,
        "mcore_bridge.model.modules.gated_delta_net",
        GatedDeltaNet=bridge_gdn,
    )
    return bridge_gdn


@pytest.fixture
def runtime(monkeypatch):
    def create(config=_NativeConfig, version="0.19.0", helpers=(True, True)):
        # Patching a fresh subclass leaves the source fixtures unchanged.
        config = type("TransformerConfig", (config,), {})
        native = _module(monkeypatch, "megatron.core.ssm.gated_delta_net")
        for name, available in zip(("tensor_a2a_cp2hp", "tensor_a2a_hp2cp"), helpers):
            if available:
                setattr(native, name, lambda *args, **kwargs: None)
        transformer = _module(
            monkeypatch, "megatron.core.transformer", TransformerConfig=config
        )
        ssm = _module(monkeypatch, "megatron.core.ssm", gated_delta_net=native)
        core = _module(
            monkeypatch,
            "megatron.core",
            __version__=version,
            transformer=transformer,
            ssm=ssm,
        )
        _module(monkeypatch, "megatron", core=core)
        return config, native, _bridge(monkeypatch)

    return create


def _config(config_type, *, cp=2, heads=4, model="qwen4_exp", bridge=True):
    cls = type(
        "ModelConfig",
        (config_type,),
        {"__module__": "mcore_bridge.config.model_config" if bridge else __name__},
    )
    config = cls()
    config.context_parallel_size = cp
    config.num_heads = heads
    config.hf_model_type = model
    return config


def test_install_native_config_preserves_validation_and_repairs_bridge(runtime):
    config, native, bridge = runtime()
    original = config.__post_init__
    helpers = (native.tensor_a2a_cp2hp, native.tensor_a2a_hp2cp)

    gdn_cp_compat.install()
    patched = bridge.forward
    gdn_cp_compat.install()

    assert config.__post_init__ is original
    assert (native.tensor_a2a_cp2hp, native.tensor_a2a_hp2cp) == helpers
    assert bridge.forward is patched
    assert bridge().forward(16) == 8
    _config(config).__post_init__()
    with pytest.raises(AssertionError, match="head divisibility"):
        _config(config, heads=3).__post_init__()


def test_install_legacy_config_scopes_guard_and_preserves_validation(runtime):
    config, native, bridge = runtime(_LegacyConfig, "0.17.0", (False, False))

    gdn_cp_compat.install()
    patched = config.__post_init__
    gdn_cp_compat.install()

    assert config.__post_init__ is patched
    assert native.tensor_a2a_cp2hp is gdn_cp_compat.tensor_a2a_cp2hp
    assert native.tensor_a2a_hp2cp is gdn_cp_compat.tensor_a2a_hp2cp
    assert bridge().forward(16) == 8
    _config(config).__post_init__()
    for kwargs in ({"model": "other"}, {"bridge": False}):
        with pytest.raises(AssertionError, match="does not support context parallel"):
            _config(config, **kwargs).__post_init__()
    with pytest.raises(AssertionError, match="head divisibility"):
        _config(config, heads=3).__post_init__()
    with pytest.raises(AssertionError, match="positive CP"):
        _config(config, cp=0).__post_init__()


@pytest.mark.parametrize("version", ["0.17.0", "0.20.0", None])
def test_install_unknown_guardless_runtime_rejected(runtime, version):
    config, _, bridge = runtime(version=version)
    original = config.__post_init__
    with pytest.raises(RuntimeError, match="Unexpected TransformerConfig guard"):
        gdn_cp_compat.install()
    assert config.__post_init__ is original
    assert bridge.forward is _legacy_forward


@pytest.mark.parametrize("helpers", [(False, False), (True, False), (False, True)])
def test_install_guardless_runtime_missing_helpers_rejected(runtime, helpers):
    _, _, bridge = runtime(helpers=helpers)
    with pytest.raises(RuntimeError, match="guard|Incomplete native"):
        gdn_cp_compat.install()
    assert bridge.forward is _legacy_forward


@pytest.mark.parametrize("config", [_ChangedGuardConfig, _DuplicateGuardConfig])
def test_install_changed_legacy_guard_on_modern_runtime_rejected(runtime, config):
    runtime(config)
    with pytest.raises(RuntimeError, match="guard"):
        gdn_cp_compat.install()


def test_install_core019_config_preserves_native_cp_validation(monkeypatch):
    """Exercise the installed Core config, including its full native validation."""
    try:
        installed_version = version("megatron-core")
    except PackageNotFoundError:
        pytest.skip("Megatron-Core is not installed")
    if installed_version != "0.19.0":
        pytest.skip("Actual Core 0.19 config requires the supported 0.19.0 runtime")
    core = pytest.importorskip("megatron.core")
    if core.__version__ != "0.19.0":
        pytest.skip("Actual Core 0.19 config requires the supported 0.19.0 runtime")
    from megatron.core.ssm import gated_delta_net as native
    from megatron.core.transformer import TransformerConfig

    bridge = _bridge(monkeypatch)
    original = TransformerConfig.__post_init__
    helpers = (native.tensor_a2a_cp2hp, native.tensor_a2a_hp2cp)
    gdn_cp_compat.install()
    assert TransformerConfig.__post_init__ is original
    assert (native.tensor_a2a_cp2hp, native.tensor_a2a_hp2cp) == helpers
    assert bridge().forward(16) == 8

    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        context_parallel_size=2,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    TransformerConfig(**kwargs)
    kwargs.update(linear_num_key_heads=3, linear_num_value_heads=3)
    with pytest.raises(AssertionError, match="must be a multiple"):
        TransformerConfig(**kwargs)
