import json

import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    _detect_fp8_checkpoint,
)


class TestDetectFP8Checkpoint:
    def test_fp8_with_scale_inv_keys(self, tmp_path):
        index = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight_scale_inv": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is True

    def test_bf16_checkpoint(self, tmp_path):
        index = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_no_index_file(self, tmp_path):
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_single_file_fp8_checkpoint(self, tmp_path):
        from safetensors.torch import save_file

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                256, 512, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(
                2, 4, dtype=torch.float32
            ),
        }
        save_file(tensors, tmp_path / "model.safetensors")
        assert _detect_fp8_checkpoint(str(tmp_path)) is True

    def test_single_file_bf16_checkpoint(self, tmp_path):
        from safetensors.torch import save_file

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                256, 512, dtype=torch.bfloat16
            )
        }
        save_file(tensors, tmp_path / "model.safetensors")
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_empty_weight_map(self, tmp_path):
        index = {"weight_map": {}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is False
