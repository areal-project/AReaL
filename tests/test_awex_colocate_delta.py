# SPDX-License-Identifier: Apache-2.0
"""CPU round-trip tests for colocate delta (incremental) weight transfer.

These validate the migration's correctness contract: the sender-side
``DeltaTracker`` decision (mirrored from ``AwexMegatronAdapter._delta_encode``)
feeding dte's transport-free ``DeltaEngine.reconstruct`` (the core of
``AwexSGLangAdapter._delta_reconstruct``) reconstructs full + delta payloads
byte-identically against the receiver's version-chained base — no GPU or
distributed setup required.

``delta_config`` is loaded directly from its file so the test does not pull in
the full areal runtime import chain (httpx/awex), which is absent in lean CI
sandboxes; the module itself only depends on ``os`` + (lazily) ``dte``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dte")

_DC_PATH = (
    Path(__file__).resolve().parent.parent
    / "areal/v2/weight_update/awex/delta_config.py"
)


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
    """Mirror ``AwexMegatronAdapter._delta_encode`` (sender side)."""
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
    g = torch.Generator().manual_seed(seed)
    return {
        "embed.weight": torch.randn(32, 16, generator=g),
        "layer.weight": torch.randn(16, 16, generator=g),
        "layer.bias": torch.randn(16, generator=g),
    }


def test_delta_config_env_gates(dc, monkeypatch):
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
    monkeypatch.setenv("AWEX_DELTA_BYTES_RATIO", "0.33")
    assert dc.delta_bytes_ratio() == pytest.approx(0.33)
    monkeypatch.setenv("DTE_DELTA_BYTES_RATIO", "0.44")
    assert dc.delta_bytes_ratio() == pytest.approx(0.44)


def test_factories_build_dte_objects(dc, monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")
    assert hasattr(tracker, "encode") and hasattr(tracker, "seed")
    assert hasattr(engine, "reconstruct")


def test_full_then_delta_roundtrip(dc, monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    from dte.core import is_delta_payload

    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")

    # v1: first transfer -> full sync (dense); seeds sender snapshot + receiver base.
    w1 = _weights(0)
    names, tensors = _encode_like_adapter(tracker, w1, 1)
    assert not is_delta_payload(names)
    full1, masks1 = engine.reconstruct(_over_the_wire(names, tensors), 1)
    assert masks1 is None
    for k, v in w1.items():
        assert torch.equal(full1[k], v), f"full-sync mismatch on {k}"

    # v2: change two elements of one tensor -> sparse delta; others unchanged.
    w2 = {k: v.clone() for k, v in w1.items()}
    w2["layer.weight"][0, 0] += 1.5
    w2["layer.weight"][3, 7] -= 2.0
    names, tensors = _encode_like_adapter(tracker, w2, 2)
    assert is_delta_payload(names)
    # Unchanged tensors are not shipped at all.
    assert "embed.weight" not in names and "layer.bias" not in names
    full2, masks2 = engine.reconstruct(_over_the_wire(names, tensors), 2)
    assert masks2 is not None
    for k, v in w2.items():
        assert torch.equal(full2[k], v), f"delta reconstruct mismatch on {k}"


def test_live_decode_tracks_versions_without_cpu_base(dc, monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    from dte.core import is_delta_payload

    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")

    w1 = _weights(11)
    names, tensors = _encode_like_adapter(tracker, w1, 1)
    assert not is_delta_payload(names)
    assert engine.decode_for_live_apply(_over_the_wire(names, tensors), 1) is None
    assert engine.base_version == 1
    assert getattr(engine, "_base") == {}

    w2 = {k: v.clone() for k, v in w1.items()}
    w2["layer.weight"][0, 0] += 1.0
    names, tensors = _encode_like_adapter(tracker, w2, 2)
    decoded = engine.decode_for_live_apply(_over_the_wire(names, tensors), 2)
    assert decoded is not None
    assert decoded.header.base_version == 1
    assert engine.base_version == 1
    engine.commit_live_apply(decoded)
    assert engine.base_version == 2
    assert getattr(engine, "_base") == {}


def test_anchor_interval_forces_full_sync(dc, monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "1")  # full sync every 1 delta
    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")

    is_full = []
    w = _weights(1)
    for version in (1, 2, 3, 4):
        if version > 1:
            w = {k: v.clone() for k, v in w.items()}
            w["layer.weight"][0, version] += float(version)
        names, tensors = _encode_like_adapter(tracker, w, version)
        _full, masks = engine.reconstruct(_over_the_wire(names, tensors), version)
        is_full.append(masks is None)  # True == full sync
    # v1 seed(full), v2 delta, v3 anchor(full), v4 delta
    assert is_full == [True, False, True, False], is_full


def test_delta_before_base_raises(dc, monkeypatch):
    from dte.engine import DeltaChainBroken

    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    tracker = dc.make_delta_tracker()

    # Build a valid delta payload (requires a seeded tracker)...
    w1 = _weights(2)
    _encode_like_adapter(tracker, w1, 1)  # seeds snapshot at v1
    w2 = {k: v.clone() for k, v in w1.items()}
    w2["layer.weight"][1, 1] += 1.0
    names, tensors = _encode_like_adapter(tracker, w2, 2)

    # ...and feed it to a FRESH receiver that never received a full-sync base.
    fresh_engine = dc.make_delta_engine("cpu")
    with pytest.raises(DeltaChainBroken):
        fresh_engine.reconstruct(_over_the_wire(names, tensors), 2)


def test_header_constant_matches_dte(dc):
    # delta_config mirrors dte's wire header so the receiver can detect a delta
    # payload without importing dte; guard the mirror against drift.
    from dte.core import DELTA_HEADER_NAME

    assert dc.DELTA_HEADER_NAME == DELTA_HEADER_NAME


def test_payload_carries_delta(dc):
    assert dc.payload_carries_delta([dc.DELTA_HEADER_NAME, "layer.weight@delta_idx"])
    assert not dc.payload_carries_delta(["layer.weight", "layer.bias"])


def test_real_awex_channel_roundtrip(dc):
    """Delta payload survives the REAL awex group/reconstruct serialization path.

    Guards the highest-risk integration point: dte's variable-length int idx /
    1-D header tensors must round-trip losslessly through
    ``group_tensors_by_shape_and_dtype`` + ``reconstruct_tensors_from_groups``
    (cuda_ipc itself is identity within one process). Skipped where awex is not
    installed (lean CI sandboxes); runs in the full inference image.
    """
    tu = pytest.importorskip("awex.util.tensor_util")
    group = tu.group_tensors_by_shape_and_dtype
    reconstruct = tu.reconstruct_tensors_from_groups

    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")

    # v1 full sync through the real channel.
    w1 = _weights(3)
    names, tensors = _encode_like_adapter(tracker, w1, 1)
    groups, meta = group(tensors)
    full1, masks1 = engine.reconstruct(dict(zip(names, reconstruct(groups, meta))), 1)
    assert masks1 is None
    for k, v in w1.items():
        assert torch.equal(full1[k], v)

    # v2 delta through the real channel; assert the channel is lossless first.
    w2 = {k: v.clone() for k, v in w1.items()}
    w2["layer.weight"][2, 2] += 1.0
    names, tensors = _encode_like_adapter(tracker, w2, 2)
    groups, meta = group(tensors)
    recon = reconstruct(groups, meta)
    for name, orig in zip(names, tensors):
        assert torch.equal(dict(zip(names, recon))[name], orig), f"channel lost {name}"
    full2, masks2 = engine.reconstruct(dict(zip(names, recon)), 2)
    assert masks2 is not None
    for k, v in w2.items():
        assert torch.equal(full2[k], v)


def test_consecutive_deltas_advance_base(dc):
    """Base must advance across many deltas, not stay pinned at the seed.

    Guards ``reconstruct`` writing each rebuilt full-shard back into the CPU base
    (dte ``reconstruct_against_base`` does this in place). A regression that left
    the base at v1 would still pass a single-delta test but corrupt v3+.
    """
    tracker = dc.make_delta_tracker()
    engine = dc.make_delta_engine("cpu")

    w = _weights(7)
    names, tensors = _encode_like_adapter(tracker, w, 1)  # v1 full sync seeds base
    engine.reconstruct(_over_the_wire(names, tensors), 1)

    for version in range(2, 8):
        w = {k: v.clone() for k, v in w.items()}
        w["layer.weight"][version % 16, (version * 3) % 16] += version * 0.5
        names, tensors = _encode_like_adapter(tracker, w, version)
        full, masks = engine.reconstruct(_over_the_wire(names, tensors), version)
        assert masks is not None, f"v{version} should be a delta"
        for k, v in w.items():
            assert torch.equal(full[k], v), f"v{version} mismatch on {k}"


def test_invert_adamw_roundtrip():
    """dte.invert_adamw recovers theta_{t-1} from a real AdamW step (CPU-only).

    This is the only CPU-verifiable piece of the inversion detector — its mcore
    traversal / DP all-reduce / convert are GPU-only. Confirms both the formula
    and the (theta_t, m, v, step, lr, wd, b1, b2, eps) argument order the
    detector (AdamWInversionDetector._reconstruct_pre_step_mcore) relies on.
    """
    from dte.core import invert_adamw

    torch.manual_seed(0)
    theta_prev = torch.randn(512, dtype=torch.float32)
    p = theta_prev.clone().requires_grad_(True)
    lr, wd, b1, b2, eps = 1e-3, 0.01, 0.9, 0.999, 1e-8
    opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    p.grad = torch.randn_like(p)
    opt.step()  # theta_prev -> theta_t, leaving exp_avg / exp_avg_sq resident
    theta_t = p.detach().clone()
    st = opt.state[p]
    recovered = invert_adamw(
        theta_t, st["exp_avg"], st["exp_avg_sq"], float(st["step"]), lr, wd, b1, b2, eps
    )
    torch.testing.assert_close(recovered, theta_prev, rtol=1e-3, atol=1e-4)


def _load_delta_detect(monkeypatch):
    """Load delta_detect.py without the full areal runtime.

    It does ``from areal.utils import logging`` at module top, which would pull
    in areal/__init__ (httpx/awex, absent in lean sandboxes). Stub that one
    symbol so the detector logic is importable for CPU tests.
    """
    import importlib.util
    import logging as _stdlog
    import sys
    import types

    fake = types.ModuleType("areal.utils.logging")
    fake.getLogger = _stdlog.getLogger
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", types.ModuleType("areal.utils"))
    monkeypatch.setitem(sys.modules, "areal.utils.logging", fake)
    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/delta_detect.py"
    )
    spec = importlib.util.spec_from_file_location("awex_delta_detect", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _stub_colocate_device(monkeypatch):
    import sys
    import types

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


def _load_sglang_adapter(monkeypatch):
    """Load sglang_adapter.py with only the AReaL symbols it imports stubbed."""
    import importlib.util
    import logging as _stdlog
    import sys
    import types

    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.sglang_sharding")

    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.v2", types.ModuleType("areal.v2"))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update",
        types.ModuleType("areal.v2.weight_update"),
    )

    awex_mod = types.ModuleType("areal.v2.weight_update.awex")
    awex_mod.awex_wu_use_group = lambda: False
    awex_mod.fetch_kv_metadata = lambda *args, **kwargs: None
    awex_mod.load_kv_metadata_file = lambda *args, **kwargs: None
    awex_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "areal.v2.weight_update.awex", awex_mod)
    _stub_colocate_device(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")
    delta_config_mod.delta_transfer_enabled = lambda: False
    delta_config_mod.make_delta_engine = lambda *args, **kwargs: None
    delta_config_mod.payload_carries_delta = lambda names: False
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

    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = _stdlog.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/sglang_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("awex_sglang_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_megatron_adapter(monkeypatch):
    """Load megatron_adapter.py with runtime-heavy AReaL imports stubbed."""
    import importlib.util
    import logging as _stdlog
    import sys
    import types

    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.param_sharding")
    pytest.importorskip("awex.transfer.transfer_plan")
    pytest.importorskip("awex.util.tensor_util")

    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.v2", types.ModuleType("areal.v2"))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update",
        types.ModuleType("areal.v2.weight_update"),
    )

    awex_mod = types.ModuleType("areal.v2.weight_update.awex")
    awex_mod.awex_wu_use_group = lambda: False
    awex_mod.fetch_kv_metadata = lambda *args, **kwargs: None
    awex_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "areal.v2.weight_update.awex", awex_mod)
    _stub_colocate_device(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")
    delta_config_mod.delta_transfer_enabled = lambda: True
    delta_config_mod.make_delta_tracker = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    delta_detect_mod = types.ModuleType("areal.v2.weight_update.awex.delta_detect")
    delta_detect_mod.build_detector = lambda *args, **kwargs: None
    delta_detect_mod.delta_detector_mode = lambda: "inversion"
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

    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = _stdlog.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/megatron_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("awex_megatron_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_initial_full_sync_uses_dense_payload_with_awex_converter(monkeypatch):
    """Initial inversion frame is dense; execution chooses the IPC grouping."""
    mod = _load_megatron_adapter(monkeypatch)

    class _Tracker:
        def __init__(self):
            self.seed_args = None

        def full_sync_reason(self, version):
            assert version == 1
            return "initial_full"

        def seed(self, params_list, version, store_snapshot=False):
            self.seed_args = (params_list, version, store_snapshot)

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._delta_tracker = _Tracker()
    adapter._delta_detector = type("_Detector", (), {"name": "inversion"})()
    adapter._awex_weight_converter = object()

    params = {"w": torch.arange(4)}
    names, tensors, zero_copy_full_payload = adapter._delta_encode(params, version=1)

    assert names == ["w"]
    assert tensors == [params["w"]]
    assert zero_copy_full_payload is True
    assert adapter._delta_tracker.seed_args == (list(params.items()), 1, False)


def test_bounded_full_ipc_payload_uses_independent_storage(monkeypatch):
    """Bounded full payload must use compact exporter-owned storage."""
    mod = _load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    live = torch.arange(4, dtype=torch.float32)
    owned = torch.arange(4, dtype=torch.float32) + 10
    adapter._live_module_storage_ptrs = lambda: {live.untyped_storage().data_ptr()}
    monkeypatch.setenv("DTE_COLOCATE_FULL_GROUP_MAX_BYTES", str(32))

    groups, metadata = adapter._full_tensors_for_ipc([live, owned])

    assert len(groups) == 1
    assert groups[0].data_ptr() not in {live.data_ptr(), owned.data_ptr()}
    assert metadata[0]["group_index"] == 0
    assert metadata[1]["group_index"] == 0
    rebuilt = []
    for meta in metadata:
        start = meta["offset"]
        end = start + meta["size"]
        rebuilt.append(
            groups[meta["group_index"]].view(-1)[start:end].view(meta["shape"])
        )
    assert torch.equal(rebuilt[0], live)
    assert torch.equal(rebuilt[1], owned)


def test_bounded_full_ipc_payload_respects_group_cap(monkeypatch):
    """Same-shape tensors are packed only up to the configured byte cap."""
    mod = _load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._live_module_storage_ptrs = lambda: set()
    monkeypatch.setenv("DTE_COLOCATE_FULL_GROUP_MAX_BYTES", str(32))

    tensors = [torch.full((4,), i, dtype=torch.float32) for i in range(3)]
    groups, metadata = adapter._full_tensors_for_ipc(tensors)

    assert [g.numel() for g in groups] == [8, 4]
    assert [m["group_index"] for m in metadata] == [0, 0, 1]


def test_bounded_full_ipc_payload_handles_empty_tensors(monkeypatch):
    """CUDA IPC cannot export a zero-sized storage; use a dummy group."""
    mod = _load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    empty = torch.empty(0, 3)
    adapter._live_module_storage_ptrs = lambda: set()

    groups, metadata = adapter._full_tensors_for_ipc([empty])

    assert groups[0].numel() == 1
    assert metadata[0]["shape"] == empty.shape
    assert metadata[0]["size"] == 0


def test_inversion_detector_dispatch_and_gating(monkeypatch):
    """Detector dispatch + inversion gating (CPU).

    The mcore reconstruction / DP all-reduce / convert are GPU-only and can't be
    unit-tested here; this covers the dispatch and the gating fallbacks that
    decide snapshot-vs-inversion and inversion-vs-dense.
    """
    dd = _load_delta_detect(monkeypatch)

    # snapshot detector is a no-op marker -> None (dte snapshot path downstream)
    snap = dd.build_detector("snapshot", None)
    assert snap.name == "snapshot"
    assert snap.compute_masks(["a"], [torch.zeros(4)], 1) is None
    # unknown / empty mode also falls back to snapshot
    assert dd.build_detector("", None).name == "snapshot"

    # inversion detector builds, but with no optimizers the hard gate fails and
    # compute_masks returns None -> writer ships a dense full sync that step.
    class _NoOptAdapter:
        def _get_inner_optimizers(self):
            return []

    inv = dd.build_detector("inversion", _NoOptAdapter())
    assert inv.name == "inversion"
    assert inv.compute_masks(["a"], [torch.zeros(4)], 1) is None


def test_bf16_boundary_mask_uses_asymmetric_neighbors(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    cur = torch.tensor([1.0], dtype=torch.bfloat16)
    lower = torch.nextafter(cur, torch.full_like(cur, float("-inf"))).to(torch.float32)
    lower_half_ulp = (cur.to(torch.float32) - lower) * 0.5
    old_inside_current_bin = cur.to(torch.float32) - lower_half_ulp * 0.99995

    assert old_inside_current_bin.to(torch.bfloat16).item() == cur.item()
    assert dd._bf16_rounding_boundary_mask(old_inside_current_bin, cur).item()


def test_inversion_detector_computes_masks_from_adamw_state(monkeypatch):
    """Exercise the inversion detector's real reconstruct -> mask path.

    This uses a tiny fake adapter/optimizer that exposes the Megatron distributed
    optimizer fields the detector consumes, but avoids launching mcore or
    torch.distributed. The reconstructed pre-step tensor is converted through
    the adapter override path and compared against the live tensor, which is the
    core inversion logic used by the AWEX colocate sender.
    """
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    torch.manual_seed(1)
    theta_prev = torch.randn(64, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    lr, wd, b1, b2, eps = 3e-3, 0.02, 0.9, 0.999, 1e-8
    base_opt = torch.optim.AdamW(
        [param],
        lr=lr,
        betas=(b1, b2),
        eps=eps,
        weight_decay=wd,
    )
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()
    offloaded_state = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v
        for k, v in base_opt.state[param].items()
    }
    group_step = offloaded_state.pop("step")
    base_opt.param_groups[0]["step"] = group_step
    base_opt.state[param]["exp_avg"] = torch.empty(0)
    base_opt.state[param]["exp_avg_sq"] = torch.empty(0)
    if isinstance(base_opt.state[param].get("step"), torch.Tensor):
        base_opt.state[param]["step"] = torch.empty(0)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {param: offloaded_state}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {"w": theta_by_id.get(id(param), param.detach())}

    inv = dd.build_detector("inversion", _FakeAdapter())
    # Simulate that version 1 synced the pre-step weights. AdamW inversion is
    # only valid for the next optimizer step relative to that synced watermark.
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)
    masks = inv.compute_masks(["w"], [theta_t], version=2)

    assert masks is not None
    assert set(masks) == {"w"}
    assert torch.equal(
        masks["w"], bitwise_changed_mask(theta_t, theta_prev).reshape(-1)
    )
    assert masks["w"].any()

    # After version 2 is synced, reusing the same AdamW moments without another
    # optimizer step must not replay the old update and report false positives.
    inv.mark_synced(version=2)
    same_step_masks = inv.compute_masks(["w"], [theta_t], version=3)
    assert same_step_masks is not None
    assert not same_step_masks["w"].any()

    # If the optimizer step counter advances but the synced-payload fingerprint
    # is missing, old moments must not be replayed as a fake weight delta.
    # The detector cannot prove whether weights changed, so it asks the sender
    # to do a dense fallback for this step.
    base_opt.param_groups[0]["step"] = group_step + 1
    inv._last_synced_fingerprints.clear()
    missing_fingerprint_masks = inv.compute_masks(["w"], [theta_t], version=4)
    assert missing_fingerprint_masks is None


def test_inversion_detector_uses_lr_from_completed_optimizer_step(monkeypatch):
    """LR scheduler may update param_group['lr'] before weight sync runs."""
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    torch.manual_seed(21)
    theta_prev = torch.randn(64, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    step_lr = 3e-3
    next_lr = 9e-3
    base_opt = torch.optim.AdamW(
        [param],
        lr=step_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.02,
    )
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()
    base_opt.param_groups[0]["_areal_last_step_lr"] = step_lr
    base_opt.param_groups[0]["lr"] = next_lr

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {"w": theta_by_id.get(id(param), param.detach())}

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)
    masks = inv.compute_masks(["w"], [theta_t], version=2)

    assert masks is not None
    assert torch.equal(
        masks["w"], bitwise_changed_mask(theta_t, theta_prev).reshape(-1)
    )


def test_inversion_detector_falls_back_when_non_adamw_payload_changes(monkeypatch):
    """Payload tensors not owned by AdamW cannot be inverted safely."""
    dd = _load_delta_detect(monkeypatch)

    torch.manual_seed(22)
    theta_prev = torch.randn(32, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    aux = torch.nn.Parameter(torch.randn(8, dtype=torch.float32), requires_grad=False)
    aux_synced = aux.detach().clone()

    base_opt = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    base_opt.step()
    aux.data[0] += 1.0

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _iter_model_params_for_delta(self):
            return [param, aux]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {
                "w": theta_by_id.get(id(param), param.detach()),
                "expert_bias": theta_by_id.get(id(aux), aux.detach()),
            }

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)
    inv._last_synced_payload_fingerprints["expert_bias"] = dd._tensor_fingerprint(
        aux_synced
    )

    masks = inv.compute_masks(
        ["w", "expert_bias"], [param.detach(), aux.detach()], version=2
    )
    assert masks is None


def test_inversion_detector_allows_unchanged_non_adamw_payload(monkeypatch):
    """Unchanged non-AdamW payload tensors must not force a dense fallback."""
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    torch.manual_seed(23)
    theta_prev = torch.randn(32, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    aux = torch.nn.Parameter(torch.randn(8, dtype=torch.float32), requires_grad=False)
    aux_synced = aux.detach().clone()

    base_opt = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _iter_model_params_for_delta(self):
            return [param, aux]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {
                "w": theta_by_id.get(id(param), param.detach()),
                "expert_bias": theta_by_id.get(id(aux), aux.detach()),
            }

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)
    inv._last_synced_payload_fingerprints["expert_bias"] = dd._tensor_fingerprint(
        aux_synced
    )

    masks = inv.compute_masks(["w", "expert_bias"], [theta_t, aux.detach()], version=2)

    assert masks is not None
    assert torch.equal(
        masks["w"], bitwise_changed_mask(theta_t, theta_prev).reshape(-1)
    )
    assert masks["w"].any()
    assert not masks["expert_bias"].any()


def test_inversion_detector_densifies_changed_non_adamw_payload(monkeypatch):
    """Changed non-AdamW payload with no reconstructed old tensor must go dense."""
    dd = _load_delta_detect(monkeypatch)

    torch.manual_seed(29)
    theta_prev = torch.randn(32, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    aux_prev = torch.randn(8, dtype=torch.float32)
    aux = torch.nn.Parameter(aux_prev.clone() + 1.0, requires_grad=False)

    base_opt = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _iter_model_params_for_delta(self):
            return [param, aux]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {
                "w": theta_by_id.get(id(param), param.detach()),
                "expert_bias": theta_by_id.get(id(aux), aux.detach()),
            }

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)
    inv._last_synced_payload_fingerprints["expert_bias"] = dd._tensor_fingerprint(
        aux_prev
    )

    assert inv.compute_masks(["w", "expert_bias"], [theta_t, aux.detach()], 2) is None


def test_inversion_detector_does_not_gate_one_step_on_fingerprint(monkeypatch):
    """A fingerprint collision must not hide a valid one-step AdamW update."""
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    torch.manual_seed(11)
    theta_prev = torch.randn(128, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    base_opt = torch.optim.AdamW(
        [param],
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()

    # Force the synced/current fingerprints to collide. A correct detector still
    # reconstructs a one-step AdamW update from moments instead of trusting this
    # compact fingerprint as a proof of no change.
    monkeypatch.setattr(dd, "_tensor_fingerprint", lambda _tensor: ("collision",))

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {"w": theta_by_id.get(id(param), param.detach())}

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = ("collision",)

    masks = inv.compute_masks(["w"], [theta_t], version=2)

    assert masks is not None
    assert torch.equal(
        masks["w"], bitwise_changed_mask(theta_t, theta_prev).reshape(-1)
    )
    assert masks["w"].any()


def test_inversion_detector_zero_grad_step_uses_fingerprint_guard(monkeypatch):
    """Zero current grad can still be a valid AdamW delta if the shard changed."""
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    torch.manual_seed(2)
    param = torch.nn.Parameter(torch.randn(32, dtype=torch.float32))
    base_opt = torch.optim.AdamW(
        [param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_synced = param.detach().clone()

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            return {"w": theta_by_id.get(id(param), param.detach())}

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv.mark_synced(version=2)

    param.grad = torch.zeros_like(param)
    base_opt.step()
    theta_after_zero_grad = param.detach().clone()
    assert not torch.equal(theta_after_zero_grad, theta_synced)

    masks = inv.compute_masks(["w"], [theta_after_zero_grad], version=3)
    assert masks is not None
    assert torch.equal(
        masks["w"],
        bitwise_changed_mask(theta_after_zero_grad, theta_synced).reshape(-1),
    )
    assert masks["w"].any()


def test_inversion_fingerprint_detects_value_changes(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    before = torch.arange(128, dtype=torch.float32)
    after = before.clone()
    after[17] += 1

    assert dd._tensor_fingerprint(before) != dd._tensor_fingerprint(after)
    assert dd._tensor_fingerprint(before) == dd._tensor_fingerprint(before.clone())


def test_inversion_reconstruct_missing_local_state_joins_dp_allreduce(monkeypatch):
    """A DP rank without local moments must still join per-param collectives.

    In Megatron's distributed optimizer a rank can see the full bf16 model
    param while not owning a usable optimizer-state shard for that param. The
    detector must contribute zeros and enter the same all-reduces as ranks that
    do own a shard; otherwise colocate v2 can hang waiting for one rank.
    """
    dd = _load_delta_detect(monkeypatch)

    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    base_opt = torch.optim.AdamW([param], lr=1e-3)
    dp_group = object()
    calls = []

    def _fake_all_reduce(tensor, op=None, group=None):
        del op
        assert group is dp_group
        calls.append((tuple(tensor.shape), tensor.dtype))
        if tensor.dtype == torch.int64:
            # Simulate another DP rank contributing this param's optimizer shard.
            assert tuple(tensor.shape) == (3,)
            tensor[0] = 1
            tensor[1] = 0
            tensor[2] = param.numel()

    monkeypatch.setattr(dd.torch.distributed, "all_reduce", _fake_all_reduce)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = dp_group

        def _get_model_param_range_map(self, model_param):
            assert model_param is param

            class _Range:
                start = 0
                end = param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

    inv = dd.build_detector("inversion", _FakeAdapter())
    theta_old = inv._reconstruct_pre_step_mcore([_FakeDistOpt()])

    assert theta_old is not None
    assert id(param) in theta_old
    assert torch.equal(theta_old[id(param)], param.detach())
    assert calls == [((param.numel(),), torch.float32), ((3,), torch.int64)]


def test_inversion_allows_multiple_optimizer_dp_topologies(monkeypatch):
    """MoE models can mix regular DP params with smaller expert-DP params."""
    dd = _load_delta_detect(monkeypatch)

    torch.manual_seed(33)
    dense_prev = torch.randn(16, dtype=torch.float32)
    expert_prev = torch.randn(12, dtype=torch.float32)
    dense_param = torch.nn.Parameter(dense_prev.clone())
    expert_param = torch.nn.Parameter(expert_prev.clone())
    dense_opt = torch.optim.AdamW([dense_param], lr=2e-3)
    expert_opt = torch.optim.AdamW([expert_param], lr=4e-3)
    dense_param.grad = torch.randn_like(dense_param)
    expert_param.grad = torch.randn_like(expert_param)
    dense_opt.step()
    expert_opt.step()

    dense_group = object()
    expert_group = object()
    group_signature = {
        id(dense_group): (0, 16),
        id(expert_group): (0, 2),
    }
    calls = []

    def _fake_rank(group=None):
        return group_signature.get(id(group), (-1, 1))[0]

    def _fake_world_size(group=None):
        return group_signature.get(id(group), (-1, 1))[1]

    def _fake_all_reduce(tensor, op=None, group=None):
        del op
        calls.append((group, tuple(tensor.shape), tensor.dtype))

    monkeypatch.setattr(dd, "_dist_rank", _fake_rank)
    monkeypatch.setattr(dd, "_dist_world_size", _fake_world_size)
    monkeypatch.setattr(dd.torch.distributed, "all_reduce", _fake_all_reduce)

    class _FakeDistOpt:
        model_param_group_index_map = None

        def __init__(self, base_opt, param, dp_group):
            self.optimizer = base_opt
            self.shard_fp32_from_float16_groups = [[param]]
            self.model_float16_groups = [[param]]
            self.data_parallel_group = dp_group
            self._param = param

        def _get_model_param_range_map(self, model_param):
            assert model_param is self._param

            class _Range:
                start = 0
                end = model_param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}

    inv = dd.build_detector("inversion", _FakeAdapter())
    dense_key = "model.layers.0.mlp.dense_h_to_4h.weight"
    expert_key = "model.layers.0.mlp.experts.0.w1.weight"
    inv._module_param_key_maps = lambda: (
        {id(dense_param): dense_key, id(expert_param): expert_key},
        {},
    )
    for key, prev in ((dense_key, dense_prev), (expert_key, expert_prev)):
        inv._last_synced_steps[key] = 0.0
        inv._last_synced_fingerprints[key] = dd._tensor_fingerprint(prev)

    theta_old = inv._reconstruct_pre_step_mcore(
        [
            _FakeDistOpt(dense_opt, dense_param, dense_group),
            _FakeDistOpt(expert_opt, expert_param, expert_group),
        ]
    )

    assert theta_old is not None
    assert id(dense_param) in theta_old
    assert id(expert_param) in theta_old
    assert [call[0] for call in calls] == [
        dense_group,
        dense_group,
        expert_group,
        expert_group,
    ]
