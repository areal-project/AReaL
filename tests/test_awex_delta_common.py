# SPDX-License-Identifier: Apache-2.0
"""Shared Stage-3 tests and loaders for AWEX DTE delta transfer.

The runtime adapters import optional packages such as AWEX, DTE, and httpx.
These tests file-load the target modules with narrow stubs so separation logic
can be unit-tested without importing the full AReaL runtime.
"""

from __future__ import annotations

import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
_DC_PATH = _ROOT / "areal/v2/weight_update/awex/delta_config.py"


def _load_delta_config():
    spec = importlib.util.spec_from_file_location("awex_delta_config", _DC_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def dc():
    return _load_delta_config()


def _encode_like_adapter(tracker, params, version):
    """Mirror the sender-side DeltaTracker contract used by adapters."""
    params_list = list(params.items())
    reason = tracker.full_sync_reason(version)
    if reason is not None:
        tracker.seed(params_list, version)
        return list(params.keys()), list(params.values())
    encoded = tracker.encode(params_list, version)
    return encoded.names, encoded.tensors


def _over_the_wire(names, tensors):
    """Stand in for cuda_ipc serialize/deserialize: clone to detach storage."""
    return dict(zip(names, [t.clone() for t in tensors]))


def _weights(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        "embed.weight": torch.randn(32, 16, generator=generator),
        "layer.weight": torch.randn(16, 16, generator=generator),
        "layer.bias": torch.randn(16, generator=generator),
    }


def test_delta_config_env_gates(dc, monkeypatch):
    """DTE env vars must honor DTE_* overrides over legacy AWEX_* names."""
    monkeypatch.delenv("DTE_DELTA_TRANSFER", raising=False)
    monkeypatch.delenv("AWEX_DELTA_TRANSFER", raising=False)
    assert dc.delta_transfer_enabled() is False
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", "1")
    assert dc.delta_transfer_enabled() is True
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "0")
    assert dc.delta_transfer_enabled() is False
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "1")
    assert dc.delta_transfer_enabled() is True

    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "5")
    assert dc.delta_anchor_interval() == 5
    monkeypatch.setenv("DTE_DELTA_ANCHOR_INTERVAL", "7")
    assert dc.delta_anchor_interval() == 7


def test_cuda_mem_stats_mb_without_cuda_returns_sentinel(dc):
    """CPU-only hosts must not fail while reporting DTE memory telemetry."""
    alloc_mb, peak_mb = dc.cuda_mem_stats_mb()
    if torch.cuda.is_available():
        assert alloc_mb >= 0.0
        assert peak_mb >= 0.0
    else:
        assert (alloc_mb, peak_mb) == (-1.0, -1.0)
        assert dc.cuda_mem_stats_mb(reset_peak=False) == (-1.0, -1.0)


def test_factory_builds_dte_tracker(dc, monkeypatch):
    """The lazy factory should build a DTE tracker when DTE is available."""
    pytest.importorskip("dte")
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    tracker = dc.make_delta_tracker()
    assert hasattr(tracker, "encode") and hasattr(tracker, "seed")


def test_invert_adamw_roundtrip():
    """DTE AdamW inversion recovers pre-step weights on CPU."""
    pytest.importorskip("dte")
    from dte.core import invert_adamw

    torch.manual_seed(0)
    theta_prev = torch.randn(512, dtype=torch.float32)
    param = theta_prev.clone().requires_grad_(True)
    lr, wd, b1, b2, eps = 1e-3, 0.01, 0.9, 0.999, 1e-8
    opt = torch.optim.AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    param.grad = torch.randn_like(param)
    opt.step()
    state = opt.state[param]
    recovered = invert_adamw(
        param.detach().clone(),
        state["exp_avg"],
        state["exp_avg_sq"],
        float(state["step"]),
        lr,
        wd,
        b1,
        b2,
        eps,
    )
    torch.testing.assert_close(recovered, theta_prev, rtol=1e-3, atol=1e-4)


def _stub_colocate_device(monkeypatch):
    colocate_device_mod = types.ModuleType(
        "areal.v2.weight_update.awex.colocate_device"
    )
    colocate_device_mod.device_mapping_key = lambda ip, device: f"{ip}_{device}"
    colocate_device_mod.get_colocate_ip_address = lambda: "127.0.0.1"
    colocate_device_mod.get_physical_cuda_device_id = lambda local_index=None: str(
        local_index or 0
    )
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.colocate_device",
        colocate_device_mod,
    )


def _stub_areal_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", types.ModuleType("httpx"))
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.v2", types.ModuleType("areal.v2"))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update",
        types.ModuleType("areal.v2.weight_update"),
    )

    awex_mod = types.ModuleType("areal.v2.weight_update.awex")
    awex_mod.awex_wu_use_group = lambda: False
    awex_mod.fetch_kv_metadata = lambda *args, **kwargs: ([], [])
    awex_mod.load_kv_metadata_file = lambda *args, **kwargs: None
    awex_mod.resolve_physical_gpu_id = lambda *args, **kwargs: 0
    awex_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "areal.v2.weight_update.awex", awex_mod)
    _stub_colocate_device(monkeypatch)

    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = stdlib_logging.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    infra_mod = types.ModuleType("areal.infra")
    platforms_mod = types.ModuleType("areal.infra.platforms")
    platforms_mod.current_platform = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "areal.infra", infra_mod)
    monkeypatch.setitem(sys.modules, "areal.infra.platforms", platforms_mod)

    weight_digest_mod = types.ModuleType("areal.v2.weight_update.awex.weight_digest")
    weight_digest_mod.log_tensor_digest = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.weight_digest",
        weight_digest_mod,
    )


def _load_delta_detect(monkeypatch):
    """Load delta_detect.py without the full AReaL runtime."""
    fake = types.ModuleType("areal.utils.logging")
    fake.getLogger = stdlib_logging.getLogger
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", types.ModuleType("areal.utils"))
    monkeypatch.setitem(sys.modules, "areal.utils.logging", fake)
    for package in (
        "areal.v2",
        "areal.v2.weight_update",
        "areal.v2.weight_update.awex",
    ):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        _load_delta_config(),
    )
    path = _ROOT / "areal/v2/weight_update/awex/delta_detect.py"
    spec = importlib.util.spec_from_file_location("awex_delta_detect", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _bind_inversion_param_names(inv, *named_params):
    """Bind fake mcore parameter names for CPU-only inversion tests."""
    id2key = {id(param): name for name, param in named_params}
    inv._module_param_key_maps = lambda: (id2key, {})


def _load_sglang_adapter(monkeypatch):
    """Load sglang_adapter.py with only its AReaL imports stubbed."""
    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.sglang_sharding")

    _stub_areal_packages(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")
    delta_config_mod.separation_delta_transfer_enabled = lambda: False
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    inference_adapter_mod = types.ModuleType("areal.v2.weight_update.inference_adapter")

    class _AwexInferenceAdapter:
        pass

    inference_adapter_mod.AwexInferenceAdapter = _AwexInferenceAdapter
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.inference_adapter",
        inference_adapter_mod,
    )

    nccl_group_mod = types.ModuleType("areal.v2.weight_update.nccl_group")
    nccl_group_mod.init_weights_update_group = lambda *args, **kwargs: None
    nccl_group_mod.setup_batch_isend_irecv = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.nccl_group",
        nccl_group_mod,
    )

    path = _ROOT / "areal/v2/weight_update/awex/sglang_adapter.py"
    spec = importlib.util.spec_from_file_location("awex_sglang_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_megatron_adapter(monkeypatch):
    """Load megatron_adapter.py with runtime-heavy AReaL imports stubbed."""
    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.param_sharding")
    pytest.importorskip("awex.transfer.transfer_plan")
    pytest.importorskip("awex.util.tensor_util")

    _stub_areal_packages(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")
    delta_config_mod.separation_delta_transfer_enabled = lambda: True
    delta_config_mod.make_delta_tracker = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    delta_detect_mod = types.ModuleType("areal.v2.weight_update.awex.delta_detect")
    delta_detect_mod.AdamWInversionDetector = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_detect",
        delta_detect_mod,
    )

    nccl_group_mod = types.ModuleType("areal.v2.weight_update.nccl_group")
    nccl_group_mod.init_weights_update_group = lambda *args, **kwargs: None
    nccl_group_mod.setup_batch_isend_irecv = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.nccl_group",
        nccl_group_mod,
    )

    training_adapter_mod = types.ModuleType("areal.v2.weight_update.training_adapter")

    class _AwexTrainingAdapter:
        pass

    training_adapter_mod.AwexTrainingAdapter = _AwexTrainingAdapter
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.training_adapter",
        training_adapter_mod,
    )

    path = _ROOT / "areal/v2/weight_update/awex/megatron_adapter.py"
    spec = importlib.util.spec_from_file_location("awex_megatron_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
