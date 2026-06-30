"""Unit tests for LoRA disk-based weight synchronization."""

import copy
import json
import os
from dataclasses import asdict

import pytest
import torch

from areal.api import ParamSpec, WeightUpdateMeta
from areal.api.cli_args import TrainEngineConfig
from areal.api.io_struct import get_versioned_lora_name

# Keep this in sync with ``FSDPEngine._save_lora_adapter_to_hf``.
_LORA_KEYWORDS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def _make_dummy_model_params() -> dict[str, torch.Tensor]:
    """Return a dict simulating ``named_parameters`` of a LoRA-wrapped model.

    Keys follow the PEFT naming convention produced by
    ``peft.get_peft_model`` on a HuggingFace transformer (i.e. they
    contain the active adapter name segment ``.default.``).
    """
    return {
        # Base model weights (non-LoRA)
        "base_model.model.model.embed_tokens.weight": torch.randn(1000, 64),
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(
            64, 64
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.base_layer.weight": torch.randn(
            64, 64
        ),
        "base_model.model.lm_head.weight": torch.randn(1000, 64),
        # LoRA adapter weights (with PEFT ".default." segment)
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
            8, 64
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
            64, 8
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.randn(
            8, 64
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.randn(
            64, 8
        ),
    }


def _is_lora_param(name: str) -> bool:
    return any(kw in name for kw in _LORA_KEYWORDS)


def _filter_lora_adapter_state(
    params: dict[str, torch.Tensor], adapter_name: str = "default"
) -> dict[str, torch.Tensor]:
    """Replicate ``FSDPEngine._save_lora_adapter_to_hf`` filtering logic.

    Selects only LoRA tensors and strips the ``.<adapter_name>.`` segment
    so the resulting keys match the standard PEFT adapter file layout
    (e.g. ``...lora_A.weight``), which is what SGLang's
    ``/load_lora_adapter`` expects.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in params.items():
        if not _is_lora_param(name):
            continue
        stripped = name.replace(f".{adapter_name}.", ".")
        out[stripped] = tensor
    return out


# ---------------------------------------------------------------------------
# Test: LoRA adapter filtering / key normalisation
# ---------------------------------------------------------------------------


class TestLoRAAdapterFiltering:
    """Mirrors ``FSDPEngine._save_lora_adapter_to_hf`` selection logic."""

    def test_filter_returns_only_lora(self):
        params = _make_dummy_model_params()
        adapter = _filter_lora_adapter_state(params)
        assert len(adapter) == 4
        for name in adapter:
            assert _is_lora_param(name)

    def test_filter_excludes_base(self):
        params = _make_dummy_model_params()
        adapter = _filter_lora_adapter_state(params)
        for name in adapter:
            assert "base_layer" not in name

    def test_filter_strips_default_segment(self):
        """After filtering, ``.default.`` must be removed (PEFT format)."""
        params = _make_dummy_model_params()
        adapter = _filter_lora_adapter_state(params)
        for name in adapter:
            assert ".default." not in name
            assert _is_lora_param(name)
            assert name.endswith(".weight")

    def test_filter_empty_when_no_lora(self):
        params = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }
        assert _filter_lora_adapter_state(params) == {}

    def test_lora_param_shapes_match_rank(self):
        """LoRA A has shape (rank, in_features); B has (out_features, rank)."""
        params = _make_dummy_model_params()
        adapter = _filter_lora_adapter_state(params)
        lora_rank = 8
        for name, tensor in adapter.items():
            if "lora_A" in name:
                assert tensor.shape[0] == lora_rank
            elif "lora_B" in name:
                assert tensor.shape[1] == lora_rank

    def test_adapter_keys_match_load_lora_adapter_format(self):
        """The stripped keys are exactly what SGLang's
        ``/load_lora_adapter`` endpoint consumes when it reads the
        ``adapter_model.safetensors`` next to ``adapter_config.json``.
        """
        params = _make_dummy_model_params()
        adapter = _filter_lora_adapter_state(params)
        for k in adapter:
            assert k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"), k


# ---------------------------------------------------------------------------
# Test: WeightUpdateMeta schema for LoRA disk sync
# ---------------------------------------------------------------------------


class TestWeightUpdateMetaForLoRA:
    """Validate the LoRA-relevant fields on WeightUpdateMeta."""

    def test_default_flags(self):
        meta = WeightUpdateMeta(type="disk")
        assert meta.use_lora is False
        assert meta.lora_name == ""
        assert meta.lora_int_id == 0
        assert meta.version is None
        # Default peft_config must be an empty dict (not None) so
        # downstream callers can do a plain ``meta.peft_config.get(...)``
        # without a None-check.
        assert meta.peft_config == {}

    def test_construct_lora_meta(self):
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            lora_int_id=1,
            peft_config={"r": 16, "lora_alpha": 16},
            path="/tmp/weight_update",
        )
        assert meta.use_lora is True
        assert meta.lora_name == "my-lora"
        assert meta.peft_config["r"] == 16

    def test_asdict_round_trip(self):
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            lora_int_id=1,
            peft_config={"r": 16, "lora_alpha": 16},
        )
        d = asdict(meta)
        assert d["type"] == "disk"
        assert d["use_lora"] is True
        assert d["lora_name"] == "my-lora"
        assert d["peft_config"]["r"] == 16

    def test_with_version_updates_path_and_version(self):
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            path="/tmp/checkpoints/weight_update",
        )
        v3 = meta.with_version(3)
        assert v3.version == 3
        assert v3.path is not None
        assert v3.path.endswith("weight_update_v3")
        assert meta.version is None
        assert v3.use_lora is True
        assert v3.lora_name == "my-lora"

    def test_with_version_rejects_negative(self):
        meta = WeightUpdateMeta(type="disk", path="/tmp/wu")
        with pytest.raises(ValueError, match="non-negative"):
            meta.with_version(-1)

    def test_copy_preserves_lora_fields(self):
        meta = WeightUpdateMeta(
            type="disk", use_lora=True, lora_name="L", lora_int_id=42
        )
        meta_copy = copy.copy(meta)
        assert meta_copy.use_lora is True
        assert meta_copy.lora_name == "L"
        assert meta_copy.lora_int_id == 42


# ---------------------------------------------------------------------------
# Test: TrainEngineConfig
# ---------------------------------------------------------------------------


class TestTrainEngineConfigLoRA:
    """LoRA fields on TrainEngineConfig.

    Enabling LoRA disk sync only requires ``use_lora=True`` and
    ``weight_update_mode='disk'`` -- no additional flag is needed.
    """

    def test_default_use_lora_false(self):
        config = TrainEngineConfig(
            experiment_name="test", trial_name="t", backend="fsdp:d1"
        )
        assert config.use_lora is False

    def test_enable_lora_with_disk_mode(self):
        config = TrainEngineConfig(
            experiment_name="test",
            trial_name="t",
            backend="fsdp:d1",
            use_lora=True,
            lora_rank=16,
            lora_alpha=32,
            weight_update_mode="disk",
        )
        assert config.use_lora is True
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.weight_update_mode == "disk"


# ---------------------------------------------------------------------------
# Test: SGLang request-building dispatch
# ---------------------------------------------------------------------------


class TestSGLangBackendDispatch:
    """Verify ``SGLangBackend`` builds the correct HTTP requests for each
    ``WeightUpdateMeta`` shape.
    """

    def test_disk_lora_routes_to_load_lora_adapter(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="test-lora",
            version=2,
            path="/tmp/lora_weights",
        )
        requests = backend.build_disk_weight_update_requests(meta)
        assert len(requests.requests) == 1
        req = requests.requests[0]
        assert req.endpoint == "/load_lora_adapter"
        assert req.payload["lora_name"] == "test-lora-v2"
        assert req.payload["lora_path"] == "/tmp/lora_weights"

    def test_disk_full_model_routes_to_update_weights_from_disk(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(type="disk", use_lora=False, path="/tmp/full_model")
        requests = backend.build_disk_weight_update_requests(meta)
        assert len(requests.requests) == 1
        req = requests.requests[0]
        assert req.endpoint == "/update_weights_from_disk"
        assert req.payload["model_path"] == "/tmp/full_model"

    def test_disk_lora_requires_lora_name(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(type="disk", use_lora=True, version=0, path="/tmp")
        with pytest.raises(ValueError, match="LoRA name"):
            backend.build_disk_weight_update_requests(meta)

    def test_disk_lora_requires_version(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="L",
            version=None,
            path="/tmp",
        )
        with pytest.raises(ValueError, match="Version"):
            backend.build_disk_weight_update_requests(meta)

    def test_distributed_rejects_lora(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(type="xccl", use_lora=True)
        with pytest.raises(ValueError, match="does not support LoRA"):
            backend.build_distributed_weight_update_requests(meta, [])

    def test_distributed_full_model(self):
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="xccl", use_lora=False, nccl_group_name="test_group"
        )
        specs = [
            ParamSpec(
                name="model.embed_tokens.weight",
                shape=(1000, 64),
                dtype="bfloat16",
            ),
        ]
        requests = backend.build_distributed_weight_update_requests(meta, specs)
        assert len(requests.requests) == 1
        req = requests.requests[0]
        assert req.endpoint == "/update_weights_from_distributed"
        assert "model.embed_tokens.weight" in req.payload["names"]
        assert req.payload["group_name"] == "test_group"

    def test_generation_request_injects_lora_path(self):
        from areal.api import ModelRequest
        from areal.api.cli_args import GenerationHyperparameters
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        gconfig = GenerationHyperparameters(max_new_tokens=8, lora_name="my-lora")
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)
        http_req = backend.build_generation_request(req, with_lora=True, version=5)
        assert http_req.endpoint == "/generate"
        assert http_req.payload["lora_path"] == "my-lora-v5"

    def test_generation_request_without_lora(self):
        from areal.api import ModelRequest
        from areal.api.cli_args import GenerationHyperparameters
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        gconfig = GenerationHyperparameters(max_new_tokens=8)
        req = ModelRequest(input_ids=[1, 2, 3], gconfig=gconfig)
        http_req = backend.build_generation_request(req, with_lora=False, version=0)
        assert "lora_path" not in http_req.payload


# ---------------------------------------------------------------------------
# Test: get_versioned_lora_name utility
# ---------------------------------------------------------------------------


class TestGetVersionedLoraName:
    def test_basic(self):
        assert get_versioned_lora_name("lora-gsm8k", 1) == "lora-gsm8k-v1"

    def test_version_zero(self):
        assert get_versioned_lora_name("my-lora", 0) == "my-lora-v0"

    def test_version_large(self):
        assert get_versioned_lora_name("adapter", 999) == "adapter-v999"


# ---------------------------------------------------------------------------
# Test: ParamSpec
# ---------------------------------------------------------------------------


class TestParamSpecForLoRA:
    def test_construct_from_lora_tensor(self):
        spec = ParamSpec(
            name="model.layers.0.self_attn.q_proj.lora_A.weight",
            shape=(8, 64),
            dtype="bfloat16",
        )
        assert spec.name.endswith("lora_A.weight")
        assert spec.shape == (8, 64)

    def test_size_bfloat16(self):
        spec = ParamSpec(name="t", shape=(8, 64), dtype="bfloat16")
        assert spec.size == 1024  # 2 bytes * 512 elements

    def test_size_float32(self):
        spec = ParamSpec(name="t", shape=(16, 32), dtype="float32")
        assert spec.size == 2048  # 4 bytes * 512 elements


# ---------------------------------------------------------------------------
# Test: end-to-end disk-sync handshake (offline simulation)
# ---------------------------------------------------------------------------


class TestDiskSyncHandshake:
    """Simulate the end-to-end disk-mode LoRA sync without networking.

    The training side writes ``adapter_model.safetensors`` +
    ``adapter_config.json`` under ``meta.path``, and the inference side
    builds an HTTP request whose ``lora_path`` points to that same
    directory.
    """

    def test_meta_path_is_carried_into_load_lora_adapter_payload(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        # 1. Training side writes the (mock) adapter files.
        adapter_dir = tmp_path / "weight_update_v1"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump({"peft_type": "LORA", "r": 8, "lora_alpha": 16}, f)

        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            version=1,
            path=str(adapter_dir),
        )

        # 2. Inference side translates the meta into an HTTP request.
        requests = SGLangBackend().build_disk_weight_update_requests(meta)

        # 3. Verify the request points at the directory the training side
        #    just wrote.
        req = requests.requests[0]
        assert req.endpoint == "/load_lora_adapter"
        assert req.payload["lora_path"] == str(adapter_dir)
        assert os.path.exists(
            os.path.join(req.payload["lora_path"], "adapter_config.json")
        )
        assert os.path.exists(
            os.path.join(req.payload["lora_path"], "adapter_model.safetensors")
        )
        assert req.payload["lora_name"] == "my-lora-v1"

    def test_with_version_changes_path_and_lora_name_consistently(self, tmp_path):
        """Across versions, the path and the lora_name must stay aligned."""
        from areal.engine.sglang_remote import SGLangBackend

        base_meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            path=str(tmp_path / "weight_update"),
        )
        for v in [0, 1, 2, 7]:
            m = base_meta.with_version(v)
            req = SGLangBackend().build_disk_weight_update_requests(m).requests[0]
            assert req.payload["lora_name"] == f"my-lora-v{v}"
            assert req.payload["lora_path"].endswith(f"weight_update_v{v}")
