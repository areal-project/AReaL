# SPDX-License-Identifier: Apache-2.0
"""Shared Stage-3 tests and loaders for AWEX DTE delta transfer.

The runtime adapters import optional packages such as AWEX, DTE, and httpx.
These tests file-load the target modules with narrow stubs so separation logic
can be unit-tested without importing the full AReaL runtime.
"""

from __future__ import annotations

import importlib.util
import logging as stdlib_logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
_DC_PATH = _ROOT / "areal/v2/weight_update/awex/delta_config.py"


def _make_environ_stub():
    environ_mod = types.ModuleType("areal.utils.environ")

    def get_env_var(name, default=None, *, fallback_names=(), allow_empty=False):
        for candidate in (name, *fallback_names):
            value = os.environ.get(candidate)
            if value is not None and (allow_empty or value.strip() != ""):
                return value
        return default

    def get_bool_env_var(
        name,
        default="false",
        *,
        fallback_names=(),
        truthy_values=("true", "1"),
        falsy_values=("false", "0"),
        strip_value=False,
    ):
        del falsy_values
        value = get_env_var(name, default, fallback_names=fallback_names)
        value = value.strip() if strip_value else value
        return value.lower() in truthy_values

    environ_mod.get_env_var = get_env_var
    environ_mod.get_bool_env_var = get_bool_env_var
    return environ_mod


def _load_delta_config():
    spec = importlib.util.spec_from_file_location("awex_delta_config", _DC_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    module_names = ("areal", "areal.utils", "areal.utils.environ")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules["areal"] = types.ModuleType("areal")
    sys.modules["areal.utils"] = types.ModuleType("areal.utils")
    sys.modules["areal.utils.environ"] = _make_environ_stub()
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
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
    assert dc.DTERuntimeConfig.from_env().delta_transfer is False
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", "1")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is True
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "0")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is False
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "1")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is True

    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "5")
    assert dc.DTERuntimeConfig.from_env().anchor_interval == 5
    monkeypatch.setenv("DTE_DELTA_ANCHOR_INTERVAL", "7")
    assert dc.DTERuntimeConfig.from_env().anchor_interval == 7


def test_delta_runtime_config_preserves_legacy_values_and_snapshots(dc, monkeypatch):
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "")
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", " yes ")
    monkeypatch.setenv("DTE_SEPARATION_WEIGHT_UPDATE", "on")

    config = dc.DTERuntimeConfig.from_env()

    assert config.delta_transfer is True
    assert config.separation_weight_update is True
    assert config.enabled is True

    monkeypatch.setenv("DTE_DELTA_TRANSFER", "0")
    assert config.enabled is True
    assert dc.DTERuntimeConfig.from_env().enabled is False


def test_wire_dtype_union_is_global_sorted_and_includes_empty_local_rank(
    dc, monkeypatch
):
    """Every rank derives the same rounds even when its local plan is empty."""
    plan = SimpleNamespace(operations={})
    group = object()

    def _all_gather_object(output, local_names, group=None):
        assert local_names == []
        assert group is not None
        output[:] = [[], ["torch.float32", "torch.bfloat16"]]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    dtypes = dc.synchronize_wire_dtypes(plan, group)

    assert dtypes == (torch.bfloat16, torch.float32)


def test_wire_dtype_union_ignores_rank_local_discovery_order(dc, monkeypatch):
    """Canonical ordering is independent of each rank's metadata insertion order."""
    op = SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=torch.float32))
    plan = SimpleNamespace(operations={1: [op]})

    def _all_gather_object(output, local_names, group=None):
        assert local_names == ["torch.float32"]
        output[:] = [
            ["torch.float32", "torch.bfloat16"],
            ["torch.bfloat16", "torch.float32"],
        ]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    dtypes = dc.synchronize_wire_dtypes(plan, object())

    assert dtypes == (torch.bfloat16, torch.float32)


def test_wire_dtype_union_preserves_single_bfloat16_round(dc, monkeypatch):
    """The established BF16-only path retains exactly one protocol round."""
    op = SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=torch.bfloat16))
    plan = SimpleNamespace(operations={1: [op]})

    def _all_gather_object(output, local_names, group=None):
        assert local_names == ["torch.bfloat16"]
        output[:] = [["torch.bfloat16"], ["torch.bfloat16"]]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    assert dc.synchronize_wire_dtypes(plan, object()) == (torch.bfloat16,)


def test_wire_dtype_union_rejects_globally_empty_plan(dc, monkeypatch):
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    def _all_gather_object(output, local_names, group=None):
        output[:] = [[], []]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)

    with pytest.raises(ValueError, match="no wire dtypes"):
        dc.synchronize_wire_dtypes(SimpleNamespace(operations={}), object())


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
def test_dte_world_size_accepts_recursive_scheduler_sizes(dc, world_size):
    dc.validate_dte_world_size(world_size, world_size // 2, world_size // 2)


@pytest.mark.parametrize("world_size", [3, 5, 6])
def test_dte_world_size_rejects_incomplete_recursive_schedules(dc, world_size):
    infer_world_size = world_size // 2
    train_world_size = world_size - infer_world_size

    with pytest.raises(
        ValueError,
        match=(
            f"inference={infer_world_size}, training={train_world_size}, "
            f"combined={world_size}"
        ),
    ):
        dc.validate_dte_world_size(
            world_size,
            infer_world_size,
            train_world_size,
        )


@pytest.mark.parametrize(
    ("combined_world_size", "infer_world_size", "train_world_size"),
    [(1, 1, 0), (4, 1, 2)],
)
def test_dte_world_size_rejects_missing_side_or_inconsistent_total(
    dc, combined_world_size, infer_world_size, train_world_size
):
    with pytest.raises(ValueError, match=f"combined={combined_world_size}"):
        dc.validate_dte_world_size(
            combined_world_size,
            infer_world_size,
            train_world_size,
        )


def test_factory_builds_dte_tracker(dc, monkeypatch):
    """The lazy factory should build a DTE tracker when DTE is available."""
    pytest.importorskip("dte")
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    tracker = dc.DTERuntimeConfig.from_env().create_delta_tracker()
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
    monkeypatch.setitem(sys.modules, "areal.utils.environ", _make_environ_stub())
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


def test_cuda_mem_stats_mb_without_cuda_returns_sentinel(monkeypatch):
    """CPU-only hosts must not fail while reporting DTE memory telemetry."""
    mod = _load_delta_detect(monkeypatch)
    alloc_mb, peak_mb = mod._cuda_mem_stats_mb()
    if torch.cuda.is_available():
        assert alloc_mb >= 0.0
        assert peak_mb >= 0.0
    else:
        assert (alloc_mb, peak_mb) == (-1.0, -1.0)
        assert mod._cuda_mem_stats_mb(reset_peak=False) == (-1.0, -1.0)


def test_adamw_hparams_require_and_prefer_recorded_step_lr(monkeypatch):
    """A scheduler's next-step LR cannot silently drive AdamW inversion."""
    mod = _load_delta_detect(monkeypatch)
    param_group = {
        "lr": 2e-6,
        "_areal_last_step_lr": 3e-6,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    hparams = mod._adamw_hparams(param_group)

    assert hparams is not None
    assert hparams[0] == 3e-6
    del param_group["_areal_last_step_lr"]
    assert mod._adamw_hparams(param_group) is None


def test_missing_recorded_step_lr_forces_dense_reconstruction(monkeypatch):
    """Missing LR metadata must not fall back to the scheduler's current LR."""

    def _unexpected_invert_adamw(*args, **kwargs):
        del args, kwargs
        pytest.fail("AdamW inversion must not run without a recorded step LR")

    dte_mod = types.ModuleType("dte")
    dte_mod.__path__ = []
    dte_core_mod = types.ModuleType("dte.core")
    dte_core_mod.invert_adamw = _unexpected_invert_adamw
    dte_mod.core = dte_core_mod
    monkeypatch.setitem(sys.modules, "dte", dte_mod)
    monkeypatch.setitem(sys.modules, "dte.core", dte_core_mod)

    mod = _load_delta_detect(monkeypatch)
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    before = param.detach().clone()

    inversion = mod.AdamWInversionDetector(
        SimpleNamespace(_offloaded_optimizer_states={})
    )
    _bind_inversion_param_names(inversion, ("w", param))
    inversion._last_synced_steps = {"w": None}
    inversion._last_synced_fingerprints = {"w": mod._tensor_fingerprint(before)}

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert "_areal_last_step_lr" not in optimizer.param_groups[0]

    class _FakeDistOpt:
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def __init__(self):
            self.optimizer = optimizer

        def _get_model_param_range_map(self, model_param):
            assert model_param is param
            return {"param": SimpleNamespace(start=0, end=param.numel())}

    assert inversion._reconstruct_pre_step_mcore([_FakeDistOpt()]) is None


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

    class _DTERuntimeConfig:
        @classmethod
        def from_env(cls):
            return SimpleNamespace(enabled=False)

    delta_config_mod.DTERuntimeConfig = _DTERuntimeConfig
    delta_config_mod.synchronize_wire_dtypes = lambda *args, **kwargs: ()
    delta_config_mod.validate_dte_world_size = lambda *args, **kwargs: None
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

    class _DTERuntimeConfig:
        @classmethod
        def from_env(cls):
            return SimpleNamespace(
                enabled=True,
                create_delta_tracker=lambda: None,
            )

    delta_config_mod.DTERuntimeConfig = _DTERuntimeConfig
    delta_config_mod.synchronize_wire_dtypes = lambda *args, **kwargs: ()
    delta_config_mod.validate_dte_world_size = lambda *args, **kwargs: None
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
