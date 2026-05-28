# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageReader

from areal.utils import logging

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from areal.models.tree_attn.module_archon import TreeAttentionMeta

logger = logging.getLogger("ArchonModelBase")


@dataclass
class BaseModelArgs(ABC):
    """Base class for model arguments."""

    # Attention backend type: "sdpa" or "varlen"
    attn_type: str = "varlen"

    @staticmethod
    def _get_rope_theta(hf_config: PretrainedConfig, default: float = 10000.0) -> float:
        """Extract rope_theta from HF config, compatible with transformers v4 and v5+.

        In transformers v5+, rope_theta was moved from a direct config attribute
        into the rope_parameters dict. This helper checks both locations.
        """
        # v5+: rope_theta lives inside rope_parameters dict
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        if "rope_theta" in rope_params:
            return rope_params["rope_theta"]
        # v4: rope_theta is a direct attribute
        return getattr(hf_config, "rope_theta", default)

    @classmethod
    @abstractmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        is_critic: bool = False,
        **kwargs,
    ) -> BaseModelArgs: ...


class BaseStateDictAdapter(ABC):
    """Base class for HF <-> Archon state dict conversion.

    Args:
        model_config: HuggingFace model configuration
        hf_assets_path: Path to HF assets folder containing tokenizer, model weights, etc.
            If provided and contains model.safetensors.index.json, the index will be
            parsed to build fqn_to_index_mapping for multi-file checkpoint support.
    """

    fqn_to_index_mapping: dict[str, int] | None

    def __init__(
        self, model_config: PretrainedConfig, hf_assets_path: str | None = None
    ):
        self.model_config = model_config
        self.from_hf_map: dict[str, str | None] = {}
        self.to_hf_map: dict[str, str] = {}
        self.hf_assets_path = hf_assets_path
        self.fqn_to_index_mapping = None

        if hf_assets_path:
            self._load_safetensors_index(hf_assets_path)

    def _load_safetensors_index(self, hf_assets_path: str) -> None:
        """Load model.safetensors.index.json to support multi-file checkpoint."""
        mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
        single_file_path = os.path.join(hf_assets_path, "model.safetensors")

        try:
            with open(mapping_path) as f:
                hf_safetensors_index = json.load(f)
        except FileNotFoundError:
            if not os.path.exists(single_file_path):
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. "
                    "Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
                )
            return

        if hf_safetensors_index:
            self.fqn_to_index_mapping = {}
            for hf_key, raw_index in hf_safetensors_index["weight_map"].items():
                match = re.search(r"\d+", raw_index)
                if match:
                    self.fqn_to_index_mapping[hf_key] = int(match.group(0))

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Return HuggingFaceStorageReader to read HF checkpoint.

        Args:
            path: The path to read HF checkpoint from.
            from_quantized: Whether loading from quantized checkpoint format.
                Note: Loading from quantized format is not supported by default.

        Returns:
            HuggingFaceStorageReader instance for reading HF checkpoint.
        """
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)

    def _maybe_composite_hf_key(self, hf_key: str) -> str:
        """Return the checkpoint key to use for HF save/load operations.

        Most Archon adapters use the HuggingFace key as-is. Adapters that need to
        remap text weights into a composite namespace, such as Qwen3.5 multimodal
        checkpoints, can override this hook.
        """
        return hf_key

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def to_hf(self, archon_state_dict: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def convert_single_to_hf(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]: ...


class BaseArchonModel(nn.Module, ABC):
    """Base class for Archon models."""

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def init_weights(self) -> None:
        """Initialize model parameters."""
        ...

    @abstractmethod
    def init_buffers(self, buffer_device: torch.device | str) -> None:
        """Initialize model buffers (e.g., rope_cache)."""
        ...


__all__ = [
    "BaseModelArgs",
    "BaseStateDictAdapter",
    "BaseArchonModel",
]
