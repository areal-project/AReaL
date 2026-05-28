import json
from unittest.mock import patch

import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    _detect_fp8_checkpoint,
    _prepare_fp8_state_dict,
    dequant_fp8_state_dict,
)


class TestFP8CheckpointFlow:
    def test_full_flow_fp8_to_bf16(self, tmp_path):
        weight_key = "model.layers.0.q_proj.weight"
        scale_key = "model.layers.0.q_proj.weight_scale_inv"
        index = {
            "weight_map": {
                weight_key: "model-00001.safetensors",
                scale_key: "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        sd = {weight_key: torch.randn(256, 512, dtype=torch.bfloat16)}
        sd = _prepare_fp8_state_dict(sd, str(tmp_path))
        assert sd[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in sd

        sd[weight_key] = torch.zeros(256, 512, dtype=torch.float8_e4m3fn)
        sd[scale_key] = torch.ones(2, 4, dtype=torch.float32)

        sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)
        assert sd[weight_key].dtype == torch.bfloat16
        assert scale_key not in sd

    def test_full_flow_single_file_fp8_to_bf16(self, tmp_path):
        from safetensors.torch import save_file

        weight_key = "model.layers.0.q_proj.weight"
        scale_key = "model.layers.0.q_proj.weight_scale_inv"
        weight_fp8 = torch.randn(256, 512, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 4, dtype=torch.float32)
        save_file(
            {weight_key: weight_fp8, scale_key: scale_inv},
            tmp_path / "model.safetensors",
        )

        assert _detect_fp8_checkpoint(str(tmp_path)) is True

        sd = {weight_key: torch.empty(256, 512, dtype=torch.bfloat16)}
        sd = _prepare_fp8_state_dict(sd, str(tmp_path))
        assert sd[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in sd

        sd[weight_key] = weight_fp8
        sd[scale_key] = scale_inv

        sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)
        assert sd[weight_key].dtype == torch.bfloat16
        assert scale_key not in sd

    def test_bf16_checkpoint_passthrough(self, tmp_path):
        index = {
            "weight_map": {
                "model.layers.0.q_proj.weight": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        sd = {
            "model.layers.0.q_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16)
        }
        assert not _detect_fp8_checkpoint(str(tmp_path))
        result = dequant_fp8_state_dict(sd)
        assert result is sd
        assert result["model.layers.0.q_proj.weight"].dtype == torch.bfloat16

    def test_no_index_file_prepare_not_called(self, tmp_path):
        with patch(
            "areal.experimental.models.archon.fp8_checkpoint._prepare_fp8_state_dict"
        ) as mock_prepare:
            is_fp8 = _detect_fp8_checkpoint(str(tmp_path))
            assert not is_fp8
            mock_prepare.assert_not_called()
