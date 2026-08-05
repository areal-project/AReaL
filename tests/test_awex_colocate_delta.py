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
from types import SimpleNamespace

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


def _changed_indices(mask: torch.Tensor) -> torch.Tensor:
    return mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(torch.int32)


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


def test_cuda_mem_stats_mb_without_cuda_returns_sentinel(dc):
    # Arrange/Act: helper must not raise on CPU-only hosts.
    alloc_mb, peak_mb = dc.cuda_mem_stats_mb()
    # Assert: sentinel on CPU; non-negative MB values when CUDA is present.
    if torch.cuda.is_available():
        assert alloc_mb >= 0.0
        assert peak_mb >= 0.0
        # reset_peak=True must clamp the next peak reading down to ~alloc.
        alloc2_mb, peak2_mb = dc.cuda_mem_stats_mb(reset_peak=False)
        assert peak2_mb <= peak_mb or peak2_mb == pytest.approx(alloc2_mb, abs=1.0)
    else:
        assert (alloc_mb, peak_mb) == (-1.0, -1.0)
        assert dc.cuda_mem_stats_mb(reset_peak=False) == (-1.0, -1.0)


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
    # delta_detect imports cuda_mem_stats_mb from delta_config; inject the real
    # module (file-loaded, dependency-free) along the stubbed package chain.
    for pkg in ("areal.v2", "areal.v2.weight_update", "areal.v2.weight_update.awex"):
        monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        _load_delta_config(),
    )
    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/delta_detect.py"
    )
    spec = importlib.util.spec_from_file_location("awex_delta_detect", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _bind_inversion_param_names(inv, *named_params):
    """Bind fake mcore parameter names for CPU-only inversion tests."""
    id2key = {id(param): name for name, param in named_params}
    inv._module_param_key_maps = lambda: (id2key, {})


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
    delta_config_mod.cuda_mem_stats_mb = lambda reset_peak=True: (-1.0, -1.0)
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
    delta_config_mod.cuda_mem_stats_mb = lambda reset_peak=True: (-1.0, -1.0)
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    delta_detect_mod = types.ModuleType("areal.v2.weight_update.awex.delta_detect")
    delta_detect_mod.build_detector = lambda *args, **kwargs: None
    delta_detect_mod.delta_detector_mode = lambda: "inversion"
    delta_detect_mod.external_delta_detector_enabled = lambda mode=None: (
        mode or delta_detect_mod.delta_detector_mode()
    ) in {"inversion", "dirty_bit", "dirty-bit", "bitset", "fused_dirty_bit"}
    delta_detect_mod.dirty_bit_detector_enabled = lambda mode=None: (
        mode or delta_detect_mod.delta_detector_mode()
    ) in {"dirty_bit", "dirty-bit", "bitset", "fused_dirty_bit"}
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_detect",
        delta_detect_mod,
    )

    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = _stdlog.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    for module_name in ("delta_index_remap", "step_dirty_dryrun"):
        module_path = (
            Path(__file__).resolve().parent.parent
            / "areal"
            / "v2"
            / "weight_update"
            / "awex"
            / f"{module_name}.py"
        )
        module_spec = importlib.util.spec_from_file_location(
            f"areal.v2.weight_update.awex.{module_name}",
            module_path,
        )
        module = importlib.util.module_from_spec(module_spec)
        assert module_spec is not None
        assert module_spec.loader is not None
        module_spec.loader.exec_module(module)
        monkeypatch.setitem(
            sys.modules,
            f"areal.v2.weight_update.awex.{module_name}",
            module,
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

    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/megatron_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("awex_megatron_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_sglang_bailing_names_match_train_converter(monkeypatch):
    """Bailing colocate must use the same keys as AWEX's mcore converter."""
    mod = _load_sglang_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexSGLangAdapter)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    embed = adapter._unfuse_params("model.word_embeddings.weight", tensor)
    assert len(embed) == 1
    assert embed[0][0] == "model.embed_tokens.weight"
    assert embed[0][1].data_ptr() == tensor.data_ptr()

    fused_name = "model.layers.7.attention.fused_qkv_a_proj_with_mqa.weight"
    fused = adapter._unfuse_params(fused_name, tensor)
    assert len(fused) == 1
    assert fused[0][0] == fused_name
    assert fused[0][1].data_ptr() == tensor.data_ptr()


def test_sglang_decoded_delta_empty_detection(monkeypatch):
    """Empty live-apply deltas can skip receiver weight resume/apply."""
    mod = _load_sglang_adapter(monkeypatch)

    decoded = type(
        "_Decoded",
        (),
        {
            "sparse": {"w": (torch.empty(0, dtype=torch.int32), torch.empty(0))},
            "dense": {},
        },
    )()
    assert mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)

    decoded.sparse = {"w": (torch.tensor([1], dtype=torch.int32), torch.ones(1))}
    assert not mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)

    decoded.sparse = {}
    decoded.dense = {"w": torch.ones(1)}
    assert not mod.AwexSGLangAdapter._decoded_delta_is_empty(decoded)


def test_sglang_colocate_device_id_uses_scheduler_physical_gpu(monkeypatch):
    mod = _load_sglang_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexSGLangAdapter)
    adapter._scheduler = type("_Scheduler", (), {"gpu_id": 4})()

    assert adapter._get_colocate_device_id() == "4"


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


def test_dirty_bit_detector_initial_frame_uses_full_sync(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)

    class _Tracker:
        def __init__(self):
            self.seed_args = None

        def full_sync_reason(self, version):
            assert version == 1
            return None

        def seed(self, params_list, version, store_snapshot=False):
            self.seed_args = (params_list, version, store_snapshot)

    class _DirtyDetector:
        name = "dirty_bit"

        def has_synced_watermark(self):
            return False

        def compute_masks(self, names, tensors, version):
            raise AssertionError("initial dirty-bit frame should full sync")

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._delta_tracker = _Tracker()
    adapter._delta_detector = _DirtyDetector()
    adapter._awex_weight_converter = object()

    params = {"w": torch.arange(4)}
    names, tensors, zero_copy_full_payload = adapter._delta_encode(params, version=1)

    assert names == ["w"]
    assert tensors == [params["w"]]
    assert zero_copy_full_payload is True
    assert adapter._delta_tracker.seed_args == (list(params.items()), 1, False)


def test_delta_encode_consumes_dirty_bit_external_indices(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)

    class _Tracker:
        def __init__(self):
            self.encode_args = None

        def full_sync_reason(self, version):
            assert version == 2
            return None

        def encode(self, params_list, version, masks=None):
            self.encode_args = (params_list, version, masks)
            return type(
                "_Encoded",
                (),
                {
                    "names": ["w@delta_idx", "w@delta_val"],
                    "tensors": [
                        masks["w"],
                        params_list[0][1].reshape(-1)[masks["w"].long()],
                    ],
                    "changed_elements": int(masks["w"].numel()),
                    "total_elements": int(params_list[0][1].numel()),
                    "num_sparse": 1,
                    "num_dense_fallback": 0,
                    "num_unchanged": 0,
                    "payload_bytes": int(masks["w"].numel() * 8),
                    "dense_bytes": int(params_list[0][1].numel() * 4),
                },
            )()

    class _DirtyDetector:
        name = "dirty_bit"

        def has_synced_watermark(self):
            return True

        def compute_masks(self, names, tensors, version):
            assert names == ["w"]
            assert version == 2
            return {"w": torch.tensor([1, 3], dtype=torch.int32)}

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._delta_tracker = _Tracker()
    adapter._delta_detector = _DirtyDetector()
    adapter._awex_weight_converter = object()

    param = torch.arange(4, dtype=torch.float32)
    names, tensors, zero_copy_full_payload = adapter._delta_encode({"w": param}, 2)

    assert names == ["w@delta_idx", "w@delta_val"]
    assert torch.equal(tensors[0], torch.tensor([1, 3], dtype=torch.int32))
    assert torch.equal(tensors[1], torch.tensor([1.0, 3.0]))
    assert zero_copy_full_payload is False
    _params, _version, masks = adapter._delta_tracker.encode_args
    assert torch.equal(masks["w"], torch.tensor([1, 3], dtype=torch.int32))


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


def test_dirty_bit_detector_dispatch_and_watermark(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    class _Adapter:
        def __init__(self):
            self.cleared = []

        def _dirty_bitset_masks_from_optimizer(self, names, tensors, version):
            assert names == ["w"]
            assert version == 2
            return {"w": torch.tensor([1], dtype=torch.int32)}

        def _clear_optimizer_dirty_bitsets(self, version):
            self.cleared.append(version)

    adapter = _Adapter()
    detector = dd.build_detector("dirty_bit", adapter)

    assert detector.name == "dirty_bit"
    assert detector.has_synced_watermark() is False
    detector.mark_synced(version=1)
    assert detector.has_synced_watermark() is True
    assert adapter.cleared == [1]

    masks = detector.compute_masks(["w"], [torch.arange(3)], version=2)
    assert torch.equal(masks["w"], torch.tensor([1], dtype=torch.int32))


def test_dirty_bit_after_skipped_optimizer_step_records_empty_masks(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)
    monkeypatch.delenv("DTE_STEP_DIRTY_DRY_RUN", raising=False)
    mod.delta_transfer_enabled = lambda: True
    mod.dirty_bit_detector_enabled = lambda mode=None: True

    adapter = object.__new__(mod.AwexMegatronAdapter)

    stats = adapter.after_optimizer_step(update_successful=False, grad_norm=None)
    masks = adapter._dirty_bitset_masks_from_optimizer(
        ["w"],
        [torch.arange(4)],
        version=2,
    )

    assert stats["perf/dirty_bit_complete"] == 1.0
    assert stats["perf/dirty_bit_records"] == 0.0
    assert torch.equal(masks["w"], torch.empty(0, dtype=torch.int32))


def test_dirty_bit_provider_records_optimizer_bitsets(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)
    monkeypatch.delenv("DTE_STEP_DIRTY_DRY_RUN", raising=False)
    monkeypatch.setenv("DTE_DIRTY_BIT_PROVIDER", "optimizer")
    mod.delta_transfer_enabled = lambda: True
    mod.dirty_bit_detector_enabled = lambda mode=None: True

    record = {
        "name": "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        "packed_bitset": torch.tensor([0b0000_1010], dtype=torch.uint8),
        "shape": (2, 4),
        "shard_start": 0,
        "shard_numel": 8,
    }

    class _Provider:
        def areal_consume_dirty_bitsets(self):
            return {"records": [record], "complete": True}

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._get_inner_optimizers = lambda: [_Provider()]
    adapter._engine = type(
        "_Engine",
        (),
        {
            "tf_config": type(
                "_Config",
                (),
                {
                    "hidden_size": 4,
                    "num_attention_heads": 2,
                    "num_query_groups": 1,
                    "kv_channels": 2,
                },
            )()
        },
    )()

    stats = adapter.after_optimizer_step(update_successful=True, grad_norm=1.0)
    masks = adapter._dirty_bitset_masks_from_optimizer(
        ["model.layers.0.self_attn.o_proj.weight"],
        [torch.arange(8)],
        version=2,
    )

    assert stats["perf/dirty_bit_complete"] == 1.0
    assert stats["perf/dirty_bit_records"] == 1.0
    assert torch.equal(
        masks["model.layers.0.self_attn.o_proj.weight"],
        torch.tensor([1, 3], dtype=torch.int32),
    )


def test_dirty_bit_provider_dp_gather_fills_non_expert_masks(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)

    local_record = mod.MCoreShardDirtyBitset(
        name="module.module.decoder.layers.0.self_attention.linear_proj.weight",
        packed_bitset=torch.tensor([0b0000_0010], dtype=torch.uint8),
        shape=(4, 4),
        shard_start=0,
        shard_numel=8,
    )
    remote_record = mod.MCoreShardDirtyBitset(
        name="module.module.decoder.layers.0.self_attention.linear_proj.weight",
        packed_bitset=torch.tensor([0b0000_0100], dtype=torch.uint8),
        shape=(4, 4),
        shard_start=8,
        shard_numel=8,
    )

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._optimizer_dirty_bitsets = [local_record]
    adapter._optimizer_dirty_bitsets_complete = True
    adapter._optimizer_dirty_bitsets_version = None
    adapter._build_rank_info = lambda: SimpleNamespace(
        dp_size=2,
        ep_size=1,
        ep_rank=0,
    )
    adapter._engine = type(
        "_Engine",
        (),
        {
            "tf_config": type(
                "_Config",
                (),
                {
                    "hidden_size": 4,
                    "num_attention_heads": 2,
                    "num_query_groups": 1,
                    "kv_channels": 2,
                },
            )(),
            "hf_config": type("_HFConfig", (), {})(),
        },
    )()
    adapter._gather_non_expert_dirty_bitsets_across_dp = lambda records: list(
        records
    ) + [remote_record]

    masks = adapter._dirty_bitset_masks_from_optimizer(
        ["model.layers.0.self_attn.o_proj.weight"],
        [torch.arange(16)],
        version=2,
    )

    assert torch.equal(
        masks["model.layers.0.self_attn.o_proj.weight"],
        torch.tensor([1, 10], dtype=torch.int32),
    )


def test_dirty_bit_provider_marks_missing_inner_optimizer_incomplete(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)
    monkeypatch.delenv("DTE_STEP_DIRTY_DRY_RUN", raising=False)
    monkeypatch.setenv("DTE_DIRTY_BIT_PROVIDER", "optimizer")
    mod.delta_transfer_enabled = lambda: True
    mod.dirty_bit_detector_enabled = lambda mode=None: True

    class _Provider:
        def areal_consume_dirty_bitsets(self):
            return {"records": [], "complete": True}

    class _NoProvider:
        pass

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._get_inner_optimizers = lambda: [_Provider(), _NoProvider()]

    stats = adapter.after_optimizer_step(update_successful=True, grad_norm=1.0)

    assert stats["perf/dirty_bit_complete"] == 0.0
    assert (
        adapter._dirty_bitset_masks_from_optimizer(
            ["w"],
            [torch.arange(4)],
            version=2,
        )
        is None
    )


def test_dirty_bit_sparse_indices_verify_against_snapshot(monkeypatch):
    mod = _load_megatron_adapter(monkeypatch)

    adapter = object.__new__(mod.AwexMegatronAdapter)
    adapter._delta_tracker = type(
        "_Tracker",
        (),
        {
            "_snapshot": {"w": torch.tensor([1, 2, 3, 4], dtype=torch.int32)},
            "_snapshot_names": {"w"},
        },
    )()
    adapter._delta_detector = type("_Detector", (), {"name": "dirty_bit"})()

    cur = torch.tensor([1, 2, 30, 4], dtype=torch.int32)

    assert adapter._delta_verify_masks_against_snapshot(
        [("w", cur)],
        {"w": torch.tensor([2], dtype=torch.int32)},
        version=2,
    )
    assert not adapter._delta_verify_masks_against_snapshot(
        [("w", cur)],
        {"w": torch.empty(0, dtype=torch.int32)},
        version=2,
    )
    assert adapter._delta_verify_masks_against_snapshot(
        [("w", cur)],
        {},
        version=2,
    )


def test_bf16_boundary_mask_uses_asymmetric_neighbors(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    cur = torch.tensor([1.0], dtype=torch.bfloat16)
    lower = torch.nextafter(cur, torch.full_like(cur, float("-inf"))).to(torch.float32)
    lower_half_ulp = (cur.to(torch.float32) - lower) * 0.5
    old_inside_current_bin = cur.to(torch.float32) - lower_half_ulp * 0.99995

    assert old_inside_current_bin.to(torch.bfloat16).item() == cur.item()
    assert dd._bf16_rounding_boundary_mask(old_inside_current_bin, cur).item()


def test_inversion_defaults_router_gate_params_to_dense(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    assert dd._inversion_force_dense_param("model.layers.9.mlp.gate.weight")
    assert not dd._inversion_force_dense_param("model.layers.9.mlp.gate_proj.weight")

    monkeypatch.setenv("DTE_DELTA_INVERSION_DENSE_PARAM_SUFFIXES", "")
    assert not dd._inversion_force_dense_param("model.layers.9.mlp.gate.weight")


def test_inversion_compute_device_follows_payload_unless_forced_cpu(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    # Meta device stands in for a non-CPU payload without requiring CUDA.
    flat_meta = torch.empty(4, device="meta")
    assert dd._inversion_compute_device(flat_meta).type == "meta"
    flat_cpu = torch.zeros(4)
    assert dd._inversion_compute_device(flat_cpu) == flat_cpu.device

    monkeypatch.setenv("DTE_INVERSION_COMPUTE_ON_CPU", "1")
    assert dd._inversion_compute_device(flat_meta) == torch.device("cpu")


def test_inversion_gates_non_decoupled_fused_adam(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    class _BaseOpt:
        adam_w_mode = 0

    class _Opt:
        optimizer = _BaseOpt()

    inv = dd.build_detector("inversion", object())
    assert inv._inversion_feasible([_Opt()]) is False


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
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
    )
    assert masks["w"].numel() > 0

    # After version 2 is synced, reusing the same AdamW moments without another
    # optimizer step must not replay the old update and report false positives.
    inv.mark_synced(version=2)
    same_step_masks = inv.compute_masks(["w"], [theta_t], version=3)
    assert same_step_masks is not None
    assert same_step_masks["w"].numel() == 0

    # If the optimizer step counter advances but the synced-payload fingerprint
    # is missing, old moments must not be replayed as a fake weight delta.
    # The detector cannot prove whether weights changed, so it asks the sender
    # to do a dense fallback for this step.
    base_opt.param_groups[0]["step"] = group_step + 1
    inv._last_synced_fingerprints.clear()
    missing_fingerprint_masks = inv.compute_masks(["w"], [theta_t], version=4)
    assert missing_fingerprint_masks is None


def test_precompute_masks_cached_and_consumed_without_recompute(monkeypatch):
    """precompute_masks caches; the next compute_masks returns the cached obj."""
    dd = _load_delta_detect(monkeypatch)

    torch.manual_seed(3)
    theta_prev = torch.randn(64, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    base_opt = torch.optim.AdamW(
        [param], lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02
    )
    param.grad = torch.randn_like(param)
    base_opt.step()
    theta_t = param.detach().clone()
    group_step = base_opt.state[param].pop("step")
    base_opt.param_groups[0]["step"] = group_step

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
    _bind_inversion_param_names(inv, ("w", param))
    inv._last_synced_steps["w"] = 0.0
    inv._last_synced_fingerprints["w"] = dd._tensor_fingerprint(theta_prev)

    payload = theta_t.to(torch.bfloat16)
    assert inv.precompute_masks(["w"], [payload], version=2) is True
    cached = inv._precomputed_masks
    assert cached is not None and cached[0] == 2 and cached[1] == ("w",)
    cached_masks = cached[2]
    assert cached_masks is not None

    # The consuming call must return the exact cached object (no recompute)
    # and clear the single-slot cache.
    masks = inv.compute_masks(["w"], [payload], version=2)
    assert masks is cached_masks
    assert inv._precomputed_masks is None


def test_precomputed_masks_stale_version_or_names_discarded(monkeypatch):
    dd = _load_delta_detect(monkeypatch)
    inv = dd.build_detector("inversion", object())
    sentinel = {"w": torch.tensor([0], dtype=torch.int32)}

    inv._precomputed_masks = (2, ("w",), sentinel)
    hit, masks = inv._pop_precomputed_masks(["w"], 3)
    assert (hit, masks) == (False, None)
    assert inv._precomputed_masks is None

    inv._precomputed_masks = (2, ("w",), sentinel)
    hit, masks = inv._pop_precomputed_masks(["other"], 2)
    assert (hit, masks) == (False, None)
    assert inv._precomputed_masks is None

    inv._precomputed_masks = (2, ("w",), sentinel)
    hit, masks = inv._pop_precomputed_masks(["w"], 2)
    assert hit is True and masks is sentinel
    assert inv._precomputed_masks is None


def test_mark_synced_drops_unconsumed_precomputed_masks(monkeypatch):
    dd = _load_delta_detect(monkeypatch)
    inv = dd.build_detector("inversion", object())
    inv._precomputed_masks = (2, ("w",), {"w": torch.tensor([0])})
    inv.mark_synced(version=2, captured_state=({}, {}, {}, {}))
    assert inv._precomputed_masks is None


def test_inversion_detector_streams_old_hf_conversion(monkeypatch):
    """AdamW inversion should not require materializing the full old-HF dict."""
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    theta_prev = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    base_opt = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.01)
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

        def __init__(self):
            self.stream_calls = 0

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _iter_hf_with_overrides(self, theta_by_id):
            self.stream_calls += 1
            yield "w", theta_by_id.get(id(param), param.detach())

        def _convert_hf_with_overrides(self, theta_by_id):
            del theta_by_id
            raise AssertionError("streaming path should avoid full old-HF dict")

    adapter = _FakeAdapter()
    inv = dd.build_detector("inversion", adapter)
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)

    masks = inv.compute_masks(["w"], [theta_t], version=2)

    assert adapter.stream_calls == 1
    assert masks is not None
    assert torch.equal(
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
    )


def test_inversion_stream_survives_adapter_consuming_theta_old(monkeypatch):
    """Adapters pop theta_old entries during streaming; masks must not change.

    The real megatron adapter's ``_iter_hf_with_overrides`` consumes
    ``theta_by_id`` (pops each entry after conversion) so reconstructed fp32
    tensors free incrementally. The detector must tolerate the dict draining
    under it and still log/compute based on the pre-drain count.
    """
    dd = _load_delta_detect(monkeypatch)
    from dte.core import bitwise_changed_mask

    theta_prev = torch.linspace(-2.0, 2.0, 48, dtype=torch.float32)
    param = torch.nn.Parameter(theta_prev.clone())
    base_opt = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.01)
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

        def _iter_hf_with_overrides(self, theta_by_id):
            # Mirror the real adapter: pop the override once converted.
            theta = theta_by_id.pop(id(param), param.detach())
            yield "w", theta

    adapter = _FakeAdapter()
    inv = dd.build_detector("inversion", adapter)
    inv._last_synced_steps[id(param)] = 0.0
    inv._last_synced_fingerprints[id(param)] = dd._tensor_fingerprint(theta_prev)

    masks = inv.compute_masks(["w"], [theta_t], version=2)

    assert masks is not None
    assert torch.equal(
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
    )


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
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
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
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
    )
    assert masks["w"].numel() > 0
    assert masks["expert_bias"].numel() == 0


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
        masks["w"],
        _changed_indices(bitwise_changed_mask(theta_t, theta_prev)),
    )
    assert masks["w"].numel() > 0


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
        _changed_indices(bitwise_changed_mask(theta_after_zero_grad, theta_synced)),
    )
    assert masks["w"].numel() > 0


def test_inversion_detector_zero_grad_bf16_unchanged_fastpath(monkeypatch):
    """A zero-grad AdamW step can update fp32 while leaving BF16 payload empty."""
    dd = _load_delta_detect(monkeypatch)

    main = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
    model_param = torch.nn.Parameter(
        main.detach().to(torch.bfloat16), requires_grad=False
    )
    synced_bf16 = model_param.detach().clone()
    base_opt = torch.optim.AdamW([main], lr=1e-3, weight_decay=0.01)
    main.grad = torch.zeros_like(main)
    base_opt.step()
    model_param.data.copy_(main.detach().to(torch.bfloat16))

    assert not torch.equal(main.detach(), synced_bf16.to(torch.float32))
    assert torch.equal(model_param.detach(), synced_bf16)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[main]]
        model_float16_groups = [[model_param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, param):
            assert param is model_param

            class _Range:
                start = 0
                end = model_param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}
        _last_optimizer_update_successful = True
        _last_optimizer_grad_norm = 0.0

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            raise AssertionError("zero fast path should not convert HF overrides")

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(model_param)] = 0.0
    inv._last_synced_fingerprints[id(model_param)] = dd._tensor_fingerprint(synced_bf16)

    masks = inv.compute_masks(["w"], [model_param.detach()], version=2)

    assert masks is not None
    assert set(masks) == {"w"}
    assert masks["w"].dtype == torch.int32
    assert masks["w"].numel() == 0


def test_inversion_zero_probe_uses_fingerprint_dirty_fast_miss(monkeypatch):
    dd = _load_delta_detect(monkeypatch)
    import dte.core

    main = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
    model_param = torch.nn.Parameter(
        main.detach().to(torch.bfloat16),
        requires_grad=False,
    )
    synced_bf16 = model_param.detach().clone()
    model_param.data[0] = torch.tensor(1.25, dtype=torch.bfloat16)
    base_opt = torch.optim.AdamW([main], lr=1e-3, weight_decay=0.01)
    base_opt.param_groups[0]["step"] = torch.tensor(1.0)

    def _fail_invert(*args, **kwargs):
        raise AssertionError("fingerprint-dirty probe should not invert AdamW")

    monkeypatch.setattr(dte.core, "invert_adamw", _fail_invert)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[main]]
        model_float16_groups = [[model_param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, param):
            assert param is model_param

            class _Range:
                start = 0
                end = model_param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}
        _last_optimizer_update_successful = True
        _last_optimizer_grad_norm = 0.0

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(model_param)] = 0.0
    inv._last_synced_fingerprints[id(model_param)] = dd._tensor_fingerprint(synced_bf16)

    assert inv._probe_bf16_payload_unchanged([_FakeDistOpt()]) is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required to cover CPU optimizer shard + CUDA payload mismatch",
)
def test_inversion_zero_fastpath_handles_cpu_main_shard_cuda_payload(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    main = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
    model_param = torch.nn.Parameter(
        main.detach().to(device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    synced_bf16 = model_param.detach().clone()
    base_opt = torch.optim.AdamW([main], lr=1e-3, weight_decay=0.01)
    main.grad = torch.zeros_like(main)
    base_opt.step()
    model_param.data.copy_(main.detach().to(device="cuda", dtype=torch.bfloat16))

    assert main.device.type == "cpu"
    assert model_param.device.type == "cuda"
    assert torch.equal(model_param.detach(), synced_bf16)

    class _FakeDistOpt:
        optimizer = base_opt
        shard_fp32_from_float16_groups = [[main]]
        model_float16_groups = [[model_param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def _get_model_param_range_map(self, param):
            assert param is model_param

            class _Range:
                start = 0
                end = model_param.numel()

            return {"param": _Range()}

    class _FakeAdapter:
        _offloaded_optimizer_states = {}
        _last_optimizer_update_successful = True
        _last_optimizer_grad_norm = 0.0

        def _get_inner_optimizers(self):
            return [_FakeDistOpt()]

        def _convert_hf_with_overrides(self, theta_by_id):
            raise AssertionError("zero fast path should not convert HF overrides")

    inv = dd.build_detector("inversion", _FakeAdapter())
    inv._last_synced_steps[id(model_param)] = 0.0
    inv._last_synced_fingerprints[id(model_param)] = dd._tensor_fingerprint(synced_bf16)

    masks = inv.compute_masks(["w"], [model_param.detach()], version=2)

    assert masks is not None
    assert masks["w"].dtype == torch.int32
    assert masks["w"].device.type == "cuda"
    assert masks["w"].numel() == 0


def test_inversion_fingerprint_detects_value_changes(monkeypatch):
    dd = _load_delta_detect(monkeypatch)

    before = torch.arange(128, dtype=torch.float32)
    after = before.clone()
    after[17] += 1

    assert dd._tensor_fingerprint(before) != dd._tensor_fingerprint(after)
    assert dd._tensor_fingerprint(before) == dd._tensor_fingerprint(before.clone())


@pytest.mark.parametrize("window_mb", ["512", "0"])
def test_inversion_reconstruct_missing_local_state_joins_dp_allreduce(
    monkeypatch, window_mb
):
    """A DP rank without local moments must still join per-param collectives.

    In Megatron's distributed optimizer a rank can see the full bf16 model
    param while not owning a usable optimizer-state shard for that param. The
    detector must contribute zeros and enter the same all-reduces as ranks that
    do own a shard; otherwise colocate v2 can hang waiting for one rank.

    Runs under both the pipelined (async window) and the synchronous
    (``DTE_INVERSION_ALLREDUCE_WINDOW_MB=0``) reduce paths; the collective
    sequence must be identical.
    """
    monkeypatch.setenv("DTE_INVERSION_ALLREDUCE_WINDOW_MB", window_mb)
    dd = _load_delta_detect(monkeypatch)

    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    base_opt = torch.optim.AdamW([param], lr=1e-3)
    dp_group = object()
    calls = []

    class _FakeWork:
        def wait(self):
            return None

    def _fake_all_reduce(tensor, op=None, group=None, async_op=False):
        del op
        assert group is dp_group
        calls.append((tuple(tensor.shape), tensor.dtype))
        if tensor.dtype in {torch.int32, torch.int64}:
            # Simulate another DP rank contributing this param's optimizer shard.
            assert tuple(tensor.shape) == (3,)
            tensor[0] = 1
            tensor[1] = 0
            tensor[2] = param.numel()
        return _FakeWork() if async_op else None

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

    class _FakeWork:
        def wait(self):
            return None

    def _fake_all_reduce(tensor, op=None, group=None, async_op=False):
        del op
        calls.append((group, tuple(tensor.shape), tensor.dtype))
        return _FakeWork() if async_op else None

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


# ---------------------------------------------------------------------------
# Precompute reuse (release critical-path opts)
# ---------------------------------------------------------------------------


def test_pop_precomputed_synced_state_sentinel_semantics(monkeypatch):
    """A cached None (infeasible) is a valid value, distinct from a miss."""
    mod = _load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    adapter._precomputed_synced_state = None
    assert adapter._pop_precomputed_synced_state(3) is mod._CAPTURE_MISS

    adapter._precomputed_synced_state = (3, None)
    assert adapter._pop_precomputed_synced_state(3) is None
    # Consumed: second pop misses.
    assert adapter._pop_precomputed_synced_state(3) is mod._CAPTURE_MISS

    state = object()
    adapter._precomputed_synced_state = (3, state)
    assert adapter._pop_precomputed_synced_state(4) is mod._CAPTURE_MISS


def test_precompute_param_sync_covers_local_only(monkeypatch):
    """Without torch.distributed the decision is purely the local marker,
    and the marker is consumed either way."""
    mod = _load_megatron_adapter(monkeypatch)
    adapter = object.__new__(mod.AwexMegatronAdapter)

    adapter._precompute_param_synced_version = 5
    assert adapter._precompute_param_sync_covers(5) is True
    assert adapter._precompute_param_synced_version is None

    adapter._precompute_param_synced_version = 5
    assert adapter._precompute_param_sync_covers(6) is False
    assert adapter._precompute_param_synced_version is None

    adapter._precompute_param_synced_version = None
    assert adapter._precompute_param_sync_covers(5) is False
