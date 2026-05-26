# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AwexFSDPAdapter colocate methods.

These tests run without GPU/distributed by mocking the engine, awex IPC
helpers, and httpx client. They verify protocol-level correctness only;
real CUDA IPC + DTensor behavior is exercised in the multi-GPU e2e test
in test_nccl_integration.py.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
import torch


@pytest.fixture
def fsdp_adapter():
    """Build an AwexFSDPAdapter with a fully mocked FSDPEngine."""
    from areal.experimental.weight_update.awex.fsdp_adapter import (
        AwexFSDPAdapter,
    )

    mock_engine = MagicMock()
    mock_engine.world_size = 2
    mock_engine.rank = 0
    mock_engine.dp_rank = 0
    mock_engine.data_parallel_world_size = 2
    mock_engine.is_vision_model = False
    mock_engine.model_config.model_type = "qwen3"
    # _iter_hf_params_local moves CPU-offloaded shards onto _engine.device;
    # in unit tests we keep tensors on CPU to avoid touching CUDA.
    mock_engine.device = torch.device("cpu")

    adapter = AwexFSDPAdapter.__new__(AwexFSDPAdapter)
    adapter._engine = mock_engine
    adapter._transfer_plan = None
    adapter._weights_update_group = None
    adapter._transfer_rank = None
    return adapter


def test_constructor_exposes_colocate_state_fields(fsdp_adapter):
    """Every colocate state field used by the four new methods must exist
    after construction so that init/execute can store into them safely."""
    # Use the real constructor path (not __new__) for this test.
    from areal.experimental.weight_update.awex.fsdp_adapter import (
        AwexFSDPAdapter,
    )

    mock_engine = MagicMock()
    adapter = AwexFSDPAdapter(mock_engine)

    assert isinstance(adapter._colocate_lock, type(threading.Lock()))
    assert adapter._colocate_http_client is None
    assert adapter._colocate_admin_api_key == "areal-admin-key"
    assert adapter._colocate_timeout_s == 120.0
    assert adapter._released_tags == set()
    assert adapter._offloaded_weights == {}


def test_init_colocate_stores_config_and_creates_http_client(fsdp_adapter):
    fsdp_adapter._colocate_lock = threading.Lock()
    fsdp_adapter._colocate_http_client = None
    fsdp_adapter._colocate_admin_api_key = "areal-admin-key"
    fsdp_adapter._colocate_timeout_s = 120.0

    fsdp_adapter.init_colocate_weight_update(
        pair_name="pair_a",
        kv_store_url="http://gw:8000",
        transfer_rank=1,
        infer_world_size=2,
        train_world_size=2,
        num_engines=1,
        master_port=29500,
        admin_api_key="custom-key",
        timeout_s=60.0,
    )

    assert fsdp_adapter._colocate_pair_name == "pair_a"
    assert fsdp_adapter._colocate_kv_store_url == "http://gw:8000"
    assert fsdp_adapter._colocate_transfer_rank == 1
    assert fsdp_adapter._colocate_infer_world_size == 2
    assert fsdp_adapter._colocate_admin_api_key == "custom-key"
    assert fsdp_adapter._colocate_timeout_s == 60.0
    assert isinstance(fsdp_adapter._colocate_http_client, httpx.Client)


def test_init_colocate_is_idempotent_for_http_client(fsdp_adapter):
    fsdp_adapter._colocate_lock = threading.Lock()
    existing = httpx.Client()
    fsdp_adapter._colocate_http_client = existing
    fsdp_adapter._colocate_admin_api_key = "areal-admin-key"
    fsdp_adapter._colocate_timeout_s = 120.0

    fsdp_adapter.init_colocate_weight_update(
        pair_name="pair_b",
        kv_store_url="http://gw:8000",
        transfer_rank=0,
        infer_world_size=1,
        train_world_size=1,
        num_engines=1,
        master_port=29500,
    )

    assert fsdp_adapter._colocate_http_client is existing
    existing.close()


def test_iter_hf_params_local_yields_local_shards_not_full_tensors(fsdp_adapter):
    """Colocate IPC must publish each rank's local DTensor shard, not the
    all-gathered full tensor — the awex transfer plan reassembles full
    tensors on the inference side via cross-engine P2P slicing."""
    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    local_shard = torch.zeros(4, 4)  # half of an [8, 4] global tensor
    plain = torch.ones(2, 3)

    mock_dtensor = MagicMock()
    mock_dtensor._local_tensor = local_shard
    mock_param_dt = MagicMock()
    mock_param_dt.data = mock_dtensor

    mock_param_plain = MagicMock()
    mock_param_plain.data = plain

    fsdp_adapter._engine.model.named_parameters.return_value = [
        ("model.layers.0.weight", mock_param_dt),
        ("model.embed_tokens.weight", mock_param_plain),
    ]
    fsdp_adapter._engine.device = torch.device("cpu")  # for unit test
    # Sentinel: must NOT be called by the local-shard path.
    fsdp_adapter._engine._get_full_tensor = MagicMock(
        side_effect=AssertionError("must not all-gather in colocate publish")
    )

    original_isinstance = isinstance

    def fake_isinstance(obj, cls):
        if cls is mod.DTensor:
            return obj is mock_dtensor
        return original_isinstance(obj, cls)

    import builtins

    builtins.isinstance = fake_isinstance
    try:
        captured = dict(fsdp_adapter._iter_hf_params_local())
    finally:
        builtins.isinstance = original_isinstance

    # The helper detaches its return value to drop autograd; storage is shared
    # but the Python object differs, so use data_ptr() to verify identity.
    assert captured["model.layers.0.weight"].data_ptr() == local_shard.data_ptr()
    assert captured["model.embed_tokens.weight"].data_ptr() == plain.data_ptr()
    fsdp_adapter._engine._get_full_tensor.assert_not_called()


def test_execute_colocate_puts_weights_then_polls_done(fsdp_adapter, monkeypatch):
    """execute_colocate must: PUT weights → poll done key → succeed."""
    fsdp_adapter._colocate_lock = threading.Lock()
    fsdp_adapter._colocate_pair_name = "p1"
    fsdp_adapter._colocate_kv_store_url = "http://gw:8000"
    fsdp_adapter._colocate_transfer_rank = 0
    fsdp_adapter._colocate_admin_api_key = "k"
    fsdp_adapter._colocate_timeout_s = 5.0
    fsdp_adapter._released_tags = set()

    plain = torch.ones(2, 3)
    mock_param = MagicMock()
    mock_param.data = plain
    fsdp_adapter._engine.model.named_parameters.return_value = [
        ("model.embed_tokens.weight", mock_param),
    ]

    captured_puts: list[tuple] = []
    poll_state = {"calls": 0}

    def mock_put(url, json=None, headers=None, timeout=None):
        captured_puts.append((url, json, headers))
        resp = MagicMock()
        resp.status_code = 200
        return resp

    def mock_get(url, timeout=None):
        poll_state["calls"] += 1
        resp = MagicMock()
        # First two polls return 404, third returns 200.
        resp.status_code = 200 if poll_state["calls"] >= 3 else 404
        if resp.status_code == 200:
            resp.json = lambda: {"value": True}
        return resp

    mock_client = MagicMock()
    mock_client.put.side_effect = mock_put
    mock_client.get.side_effect = mock_get
    fsdp_adapter._colocate_http_client = mock_client

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    monkeypatch.setattr(
        mod,
        "group_tensors_by_shape_and_dtype",
        lambda tensors: (list(tensors), [{"shape": (2, 3)}]),
    )
    monkeypatch.setattr(
        mod,
        "cuda_ipc_serialize",
        lambda payload: b"\xde\xad\xbe\xef",
    )
    # share_memory_ on a CPU tensor is a no-op for unit testing.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    fsdp_adapter.execute_colocate_weight_update(version=7)

    assert len(captured_puts) == 1
    url, payload, headers = captured_puts[0]
    assert url == "http://gw:8000/weight_meta/p1/colocate_weights_rank0_7"
    assert payload == {"value": "deadbeef"}
    assert headers == {"Authorization": "Bearer k"}

    assert poll_state["calls"] >= 3  # at least one 404 then a 200


def test_execute_colocate_raises_timeout_when_done_never_appears(
    fsdp_adapter, monkeypatch
):
    fsdp_adapter._colocate_lock = threading.Lock()
    fsdp_adapter._colocate_pair_name = "p"
    fsdp_adapter._colocate_kv_store_url = "http://gw:8000"
    fsdp_adapter._colocate_transfer_rank = 0
    fsdp_adapter._colocate_admin_api_key = "k"
    fsdp_adapter._colocate_timeout_s = 0.05
    fsdp_adapter._released_tags = set()

    fsdp_adapter._engine.model.named_parameters.return_value = []

    mock_client = MagicMock()
    mock_client.put.return_value = MagicMock(status_code=200)
    mock_client.get.return_value = MagicMock(status_code=404)
    fsdp_adapter._colocate_http_client = mock_client

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    monkeypatch.setattr(
        mod,
        "group_tensors_by_shape_and_dtype",
        lambda tensors: ([], []),
    )
    monkeypatch.setattr(mod, "cuda_ipc_serialize", lambda payload: b"")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(TimeoutError, match="did not signal completion"):
        fsdp_adapter.execute_colocate_weight_update(version=1)


def test_release_memory_offloads_dtensor_local_to_cpu(fsdp_adapter, monkeypatch):
    """release_memory(['weights']) must offload _local_tensor to CPU and
    record the original tensor for resume.

    Use MagicMock for local tensors with is_cuda=True — real CPU tensors
    cannot have is_cuda reassigned, and production code gates on is_cuda.
    """
    cpu_offloaded_dt_local = torch.zeros(2, 4)
    mock_dt_local = MagicMock()
    mock_dt_local.is_cuda = True
    mock_dt_local.dtype = torch.float32
    mock_dt_local.detach.return_value.to.return_value = cpu_offloaded_dt_local

    mock_dtensor = MagicMock()
    mock_dtensor._local_tensor = mock_dt_local
    mock_param = MagicMock()
    mock_param.data = mock_dtensor

    cpu_offloaded_plain = torch.zeros(4, 4)
    mock_plain = MagicMock()
    mock_plain.is_cuda = True
    mock_plain.dtype = torch.float32
    mock_plain.detach.return_value.to.return_value = cpu_offloaded_plain
    plain_param = MagicMock()
    plain_param.data = mock_plain

    fsdp_adapter._engine.model.named_parameters.return_value = [
        ("model.layers.0.weight", mock_param),
        ("model.embed_tokens.weight", plain_param),
    ]

    fsdp_adapter._released_tags = set()
    fsdp_adapter._offloaded_weights = {}

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    original_isinstance = isinstance

    def fake_isinstance(obj, cls):
        if cls is mod.DTensor:
            return obj is mock_dtensor
        return original_isinstance(obj, cls)

    import builtins

    builtins.isinstance = fake_isinstance
    try:
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        fsdp_adapter.release_memory(tags=["weights"])
    finally:
        builtins.isinstance = original_isinstance

    assert "weights" in fsdp_adapter._released_tags
    assert (
        fsdp_adapter._offloaded_weights["model.layers.0.weight"]
        is cpu_offloaded_dt_local
    )
    assert (
        fsdp_adapter._offloaded_weights["model.embed_tokens.weight"]
        is cpu_offloaded_plain
    )
    # DTensor: _local_tensor swapped to empty CPU
    assert mock_dtensor._local_tensor is not mock_dt_local
    # Plain: param.data swapped to empty CPU
    assert plain_param.data is not mock_plain


def test_release_memory_is_idempotent(fsdp_adapter):
    saved = torch.zeros(1)
    fsdp_adapter._released_tags = {"weights"}
    fsdp_adapter._offloaded_weights = {"x": saved}
    fsdp_adapter._engine.model.named_parameters.return_value = []

    fsdp_adapter.release_memory(tags=["weights"])

    # Already released → must not re-iterate or reset offloaded dict.
    assert fsdp_adapter._released_tags == {"weights"}
    assert fsdp_adapter._offloaded_weights == {"x": saved}


def test_release_memory_warns_on_unsupported_tags(fsdp_adapter, caplog):
    fsdp_adapter._released_tags = set()
    fsdp_adapter._offloaded_weights = {}
    fsdp_adapter._engine.model.named_parameters.return_value = []

    fsdp_adapter.release_memory(tags=["optimizer"])

    # FSDP adapter v1 only supports "weights"; "optimizer" is logged as warning
    # and ignored (not added to _released_tags).
    assert "optimizer" not in fsdp_adapter._released_tags


def test_resume_memory_restores_offloaded_dtensor_local(fsdp_adapter, monkeypatch):
    """After release+resume, _local_tensor must hold the original CUDA
    tensor (or a copy on _engine.device for our mock)."""
    saved = torch.ones(2, 4)

    mock_dtensor = MagicMock()
    mock_dtensor._local_tensor = torch.empty(0, dtype=torch.float32, device="cpu")
    mock_param = MagicMock()
    mock_param.data = mock_dtensor

    plain_saved = torch.ones(2, 3)
    plain_param = MagicMock()
    plain_param.data = torch.empty(0, dtype=torch.float32, device="cpu")

    fsdp_adapter._engine.model.named_parameters.return_value = [
        ("model.layers.0.weight", mock_param),
        ("model.embed_tokens.weight", plain_param),
    ]
    fsdp_adapter._engine.device = torch.device("cpu")  # for unit test
    fsdp_adapter._offloaded_weights = {
        "model.layers.0.weight": saved,
        "model.embed_tokens.weight": plain_saved,
    }
    fsdp_adapter._released_tags = {"weights"}

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    original_isinstance = isinstance

    def fake_isinstance(obj, cls):
        if cls is mod.DTensor:
            return obj is mock_dtensor
        return original_isinstance(obj, cls)

    import builtins

    builtins.isinstance = fake_isinstance
    try:
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        fsdp_adapter.resume_memory(tags=["weights"])
    finally:
        builtins.isinstance = original_isinstance

    assert "weights" not in fsdp_adapter._released_tags
    assert fsdp_adapter._offloaded_weights == {}
    assert torch.equal(mock_dtensor._local_tensor, saved)
    assert torch.equal(plain_param.data, plain_saved)


def test_resume_memory_is_noop_when_not_released(fsdp_adapter):
    fsdp_adapter._released_tags = set()
    fsdp_adapter._offloaded_weights = {}
    fsdp_adapter._engine.model.named_parameters.return_value = []

    fsdp_adapter.resume_memory(tags=["weights"])

    assert fsdp_adapter._released_tags == set()


def test_execute_colocate_re_releases_on_timeout_when_offloaded(
    fsdp_adapter, monkeypatch
):
    """If the transfer body raises (TimeoutError, network failure, etc.),
    release_memory must still run when weights were originally offloaded —
    otherwise _released_tags ends up out of sync with physical state."""
    fsdp_adapter._colocate_lock = threading.Lock()
    fsdp_adapter._colocate_pair_name = "p"
    fsdp_adapter._colocate_kv_store_url = "http://gw:8000"
    fsdp_adapter._colocate_transfer_rank = 0
    fsdp_adapter._colocate_admin_api_key = "k"
    fsdp_adapter._colocate_timeout_s = 0.05
    fsdp_adapter._released_tags = {"weights"}  # was offloaded before call
    fsdp_adapter._offloaded_weights = {}

    fsdp_adapter._engine.model.named_parameters.return_value = []

    call_order: list[str] = []

    def track_resume(tags=None):
        call_order.append(f"resume:{tags}")
        fsdp_adapter._released_tags.discard("weights")

    def track_release(tags=None):
        call_order.append(f"release:{tags}")
        fsdp_adapter._released_tags.add("weights")

    fsdp_adapter.resume_memory = track_resume
    fsdp_adapter.release_memory = track_release

    mock_client = MagicMock()
    mock_client.put.return_value = MagicMock(status_code=200)
    mock_client.get.return_value = MagicMock(status_code=404)  # never 200 → timeout
    fsdp_adapter._colocate_http_client = mock_client

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    monkeypatch.setattr(
        mod,
        "group_tensors_by_shape_and_dtype",
        lambda tensors: ([], []),
    )
    monkeypatch.setattr(mod, "cuda_ipc_serialize", lambda payload: b"")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(TimeoutError, match="did not signal completion"):
        fsdp_adapter.execute_colocate_weight_update(version=1)

    # Resume ran first, then release ran in finally despite the raise.
    # _released_tags ends back at {"weights"} so the offload-state invariant
    # is preserved across exceptions.
    assert call_order == ["resume:['weights']", "release:['weights']"]
    assert "weights" in fsdp_adapter._released_tags


def test_execute_colocate_resumes_then_re_releases_when_offloaded(
    fsdp_adapter, monkeypatch
):
    """If weights were already offloaded, execute_colocate must:
    resume → run transfer → release."""
    fsdp_adapter._colocate_lock = threading.Lock()
    fsdp_adapter._colocate_pair_name = "p"
    fsdp_adapter._colocate_kv_store_url = "http://gw:8000"
    fsdp_adapter._colocate_transfer_rank = 0
    fsdp_adapter._colocate_admin_api_key = "k"
    fsdp_adapter._colocate_timeout_s = 5.0
    fsdp_adapter._released_tags = {"weights"}  # already released
    fsdp_adapter._offloaded_weights = {}

    fsdp_adapter._engine.model.named_parameters.return_value = []

    call_order: list[str] = []

    def track_resume(tags=None):
        call_order.append(f"resume:{tags}")
        fsdp_adapter._released_tags.discard("weights")

    def track_release(tags=None):
        call_order.append(f"release:{tags}")
        fsdp_adapter._released_tags.add("weights")

    fsdp_adapter.resume_memory = track_resume
    fsdp_adapter.release_memory = track_release

    mock_client = MagicMock()
    mock_client.put.return_value = MagicMock(status_code=200)
    done_resp = MagicMock(status_code=200)
    done_resp.json = lambda: {"value": True}
    mock_client.get.return_value = done_resp
    fsdp_adapter._colocate_http_client = mock_client

    import areal.experimental.weight_update.awex.fsdp_adapter as mod

    monkeypatch.setattr(
        mod,
        "group_tensors_by_shape_and_dtype",
        lambda tensors: ([], []),
    )
    monkeypatch.setattr(mod, "cuda_ipc_serialize", lambda payload: b"")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    fsdp_adapter.execute_colocate_weight_update(version=3)

    assert call_order == ["resume:['weights']", "release:['weights']"]


def test_save_parameters_resumes_offloaded_weights_then_re_releases(fsdp_adapter):
    """save_parameters must temporarily resume offloaded weights for readback."""
    shard = torch.ones(2, 3)
    fsdp_adapter._released_tags = {"weights"}
    fsdp_adapter._offloaded_weights = {}

    call_order: list[str] = []

    def track_resume(tags=None):
        call_order.append(f"resume:{tags}")
        fsdp_adapter._released_tags.discard("weights")

    def track_release(tags=None):
        call_order.append(f"release:{tags}")
        fsdp_adapter._released_tags.add("weights")

    fsdp_adapter.resume_memory = track_resume
    fsdp_adapter.release_memory = track_release
    fsdp_adapter.get_local_shard_parameters = MagicMock(
        return_value={"model.embed_tokens.weight": shard}
    )

    fsdp_adapter.save_parameters("/tmp/params.pt", names=["model.embed_tokens.weight"])

    assert call_order == ["resume:['weights']", "release:['weights']"]
    fsdp_adapter.get_local_shard_parameters.assert_called_once_with(
        ["model.embed_tokens.weight"]
    )
