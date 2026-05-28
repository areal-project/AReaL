# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/state_dict_adapter.py

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

from areal.experimental.models.archon.base import BaseStateDictAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class Qwen2StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter for Qwen2 models."""

    def __init__(
        self, model_config: PretrainedConfig, hf_assets_path: str | None = None
    ):
        super().__init__(model_config, hf_assets_path)

        # HuggingFace -> Archon key mapping
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Build reverse mapping
        self.to_hf_map = {}
        for hf_key, archon_key in self.from_hf_map.items():
            if archon_key is not None:
                self.to_hf_map[archon_key] = hf_key

        self.enable_weight_tying = getattr(model_config, "tie_word_embeddings", False)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}

        for key, value in state_dict.items():
            # Skip output.weight when weight tying is enabled
            if self.enable_weight_tying and key == "output.weight":
                continue

            # Regular key mapping
            hf_key = self._convert_key_to_hf(key)
            if hf_key is not None:
                hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # Handle weight tying
        if (
            self.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
            and "model.embed_tokens.weight" in hf_state_dict
        ):
            hf_state_dict = dict(hf_state_dict)
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        state_dict = {}
        for key, value in hf_state_dict.items():
            archon_key = self._convert_key_from_hf(key)
            if archon_key is not None:
                state_dict[archon_key] = value

        return state_dict

    def convert_single_to_hf(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        # Strip activation checkpoint wrapper prefix if present
        # e.g., "layers.0._checkpoint_wrapped_module.attention.wq.weight"
        #    -> "layers.0.attention.wq.weight"
        name = name.replace("._checkpoint_wrapped_module", "")
        # Strip torch.compile wrapper prefix if present
        # e.g., "layers.0._orig_mod.attention.wq.weight"
        #    -> "layers.0.attention.wq.weight"
        name = name.replace("._orig_mod", "")

        hf_key = self._convert_key_to_hf(name)
        if hf_key is not None:
            return [(hf_key, tensor)]
        return []

    def _convert_key_to_hf(self, archon_key: str) -> str | None:
        if archon_key in self.to_hf_map:
            return self.to_hf_map[archon_key]

        match = re.search(r"layers\.(\d+)\.", archon_key)
        if match:
            layer_num = match.group(1)
            abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", archon_key)
            if abstract_key in self.to_hf_map:
                hf_abstract = self.to_hf_map[abstract_key]
                return hf_abstract.replace("{}", layer_num, 1)

        return None

    def _convert_key_from_hf(self, hf_key: str) -> str | None:
        if hf_key in self.from_hf_map:
            result = self.from_hf_map[hf_key]
            return result if result is not None else None

        match = re.search(r"layers\.(\d+)\.", hf_key)
        if match:
            layer_num = match.group(1)
            abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", hf_key)
            if abstract_key in self.from_hf_map:
                archon_abstract = self.from_hf_map[abstract_key]
                if archon_abstract is None:
                    return None
                return archon_abstract.replace("{}", layer_num, 1)

        return None
