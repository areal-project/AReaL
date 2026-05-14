"""Unit tests for ``FSDPEngine._save_lora_adapter_to_hf``.

The saved adapter must contain only LoRA tensors and PEFT metadata that
SGLang's ``/load_lora_adapter`` can consume.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file as safetensors_load_file


def _make_lora_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic state_dict in PEFT layout (``base_model.model.`` +
    ``.default.``).  Mixes base, lora_A, lora_B, and an embedding LoRA
    pair so all four LoRA keywords are exercised.
    """
    return {
        # Base weights must be filtered out.
        "base_model.model.model.embed_tokens.weight": torch.zeros(10, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros(
            4, 4
        ),
        "base_model.model.lm_head.weight": torch.zeros(10, 4),
        # LoRA tensors must be kept with ``.default.`` stripped.
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(
            8, 4
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.ones(
            4, 8
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.ones(
            8, 4
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.ones(
            4, 8
        ),
        # Embedding LoRA pair.
        "base_model.model.model.embed_tokens.lora_embedding_A.default.weight": torch.ones(
            8, 10
        ),
        "base_model.model.model.embed_tokens.lora_embedding_B.default.weight": torch.ones(
            4, 8
        ),
    }


def _make_engine_stub(
    *,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules=None,
    base_path: str = "/storage/dummy/base",
) -> SimpleNamespace:
    """Build a SimpleNamespace exposing only the attributes that
    ``_save_lora_adapter_to_hf`` reads off ``self``.
    """
    return SimpleNamespace(
        config=SimpleNamespace(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules if target_modules is not None else [],
            path=base_path,
        ),
    )


def _invoke(engine_stub, path, state_dict):
    from areal.engine.fsdp_engine import FSDPEngine

    return FSDPEngine._save_lora_adapter_to_hf(engine_stub, path, state_dict)


class TestAdapterSafetensors:
    def test_only_lora_keys_are_written(self, tmp_path):
        engine = _make_engine_stub()
        d = tmp_path / "weight_update_v0"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())

        f = d / "adapter_model.safetensors"
        assert f.exists()
        assert f.stat().st_size > 0

        loaded = safetensors_load_file(str(f))
        assert len(loaded) == 6
        for k in loaded:
            assert "base_layer" not in k
            assert "lm_head" not in k

    def test_default_segment_is_stripped(self, tmp_path):
        engine = _make_engine_stub()
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())

        loaded = safetensors_load_file(str(d / "adapter_model.safetensors"))
        for k in loaded:
            assert ".default." not in k
            assert k.endswith(".weight"), k
            assert any(
                kw in k
                for kw in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
            ), k

    def test_tensor_values_are_preserved(self, tmp_path):
        """Filtering must not mutate tensor values."""
        engine = _make_engine_stub()
        sd = _make_lora_state_dict()
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), sd)

        loaded = safetensors_load_file(str(d / "adapter_model.safetensors"))
        sample_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        assert sample_key in loaded
        torch.testing.assert_close(
            loaded[sample_key], torch.ones(8, 4), check_dtype=False
        )

    def test_missing_lora_raises(self, tmp_path):
        """Calling with a state_dict that has no LoRA params must error
        out -- this is the fail-fast guard against forgetting to wrap
        the model with PEFT.
        """
        engine = _make_engine_stub()
        d = tmp_path / "wu"
        d.mkdir()
        bare_state = {
            "model.embed_tokens.weight": torch.zeros(10, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        }
        with pytest.raises(RuntimeError, match="no LoRA adapter parameters"):
            _invoke(engine, str(d), bare_state)


class TestAdapterConfigJson:
    def _read(self, d) -> dict:
        with open(os.path.join(str(d), "adapter_config.json")) as f:
            return json.load(f)

    def test_required_fields(self, tmp_path):
        engine = _make_engine_stub(lora_rank=16, lora_alpha=32)
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())

        cfg = self._read(d)
        assert cfg["peft_type"] == "LORA"
        assert cfg["task_type"] == "CAUSAL_LM"
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["bias"] == "none"
        assert cfg["lora_dropout"] == 0.0
        assert cfg["inference_mode"] is True
        assert "target_modules" in cfg

    def test_target_modules_default_to_all_linear(self, tmp_path):
        """Empty list / ``["all-linear"]`` must serialize as the string
        ``"all-linear"`` -- which is what PEFT and SGLang both expect.
        """
        engine = _make_engine_stub(target_modules=[])
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())
        assert self._read(d)["target_modules"] == "all-linear"

        engine2 = _make_engine_stub(target_modules=["all-linear"])
        d2 = tmp_path / "wu2"
        d2.mkdir()
        _invoke(engine2, str(d2), _make_lora_state_dict())
        assert self._read(d2)["target_modules"] == "all-linear"

    def test_target_modules_explicit_list_is_preserved(self, tmp_path):
        engine = _make_engine_stub(target_modules=["q_proj", "v_proj"])
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())
        cfg = self._read(d)
        assert cfg["target_modules"] == ["q_proj", "v_proj"]

    def test_base_model_path_is_carried(self, tmp_path):
        engine = _make_engine_stub(base_path="/some/where/qwen3-0.6b")
        d = tmp_path / "wu"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())
        cfg = self._read(d)
        assert cfg["base_model_name_or_path"] == "/some/where/qwen3-0.6b"


class TestLoadLoraAdapterContract:
    """Pin the PEFT layout consumed by SGLang's LoRA loader."""

    def test_two_files_present(self, tmp_path):
        engine = _make_engine_stub()
        d = tmp_path / "weight_update_v7"
        d.mkdir()
        _invoke(engine, str(d), _make_lora_state_dict())
        files = sorted(os.listdir(str(d)))
        assert files == ["adapter_config.json", "adapter_model.safetensors"]

    def test_round_trip_size_is_stable(self, tmp_path):
        """Re-saving the same state must yield byte-identical safetensors
        (modulo timestamps, which safetensors does not embed).  This
        catches accidental ordering nondeterminism in the filter loop.
        """
        engine = _make_engine_stub()
        sd = _make_lora_state_dict()

        d1 = tmp_path / "a"
        d1.mkdir()
        _invoke(engine, str(d1), sd)
        size1 = (d1 / "adapter_model.safetensors").stat().st_size

        d2 = tmp_path / "b"
        d2.mkdir()
        _invoke(engine, str(d2), sd)
        size2 = (d2 / "adapter_model.safetensors").stat().st_size

        assert size1 == size2
