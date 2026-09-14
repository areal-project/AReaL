# SPDX-License-Identifier: Apache-2.0

"""Common utilities for testing and profiling.

This module provides utilities that are shared between tests and profiling tools.
"""

import asyncio
import os
import random
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from areal.api import InferenceEngine, RolloutWorkflow
from areal.experimental.models.archon import get_model_spec, is_supported_model
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

logger = logging.getLogger("TestingUtils")


def get_model_path(local_path: str, hf_id: str) -> str:
    """Get model path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the model from HuggingFace Hub using snapshot_download.

    Args:
        local_path: Local path to check first.
        hf_id: HuggingFace model ID to download if local not found.

    Returns:
        Path to the model (either local or downloaded).
    """
    if os.path.exists(local_path):
        logger.info(f"Model found at local path: {local_path}")
        return local_path

    try:
        logger.info(f"Downloading model from HuggingFace Hub: {hf_id}")
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            # Allow partial downloads for faster testing
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
        )
        logger.info(f"Model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download model {hf_id}: {e}")
        raise


def get_dataset_path(local_path: str, hf_id: str) -> str:
    """Get dataset path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the dataset from HuggingFace Hub using snapshot_download.

    Args:
        local_path: Local path to check first.
        hf_id: HuggingFace dataset ID to download if local not found.

    Returns:
        Path to the dataset (either local or downloaded).
    """
    if os.path.exists(local_path):
        logger.info(f"Dataset found at local path: {local_path}")
        return local_path

    try:
        logger.info(f"Downloading dataset from HuggingFace Hub: {hf_id}")
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
        )
        logger.info(f"Dataset downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download dataset {hf_id}: {e}")
        raise


# Model paths for testing (keyed by HF model_type)
# Dense models (fast to instantiate even on meta device)
DENSE_MODEL_PATHS = {
    "qwen2": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ),
    "qwen3": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/",
        "Qwen/Qwen3-0.6B",
    ),
    "qwen3_5": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3.5-0.8B/",
        "Qwen/Qwen3.5-0.8B",
    ),
    "qwen2_5_vl": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-VL-3B-Instruct/",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "qwen3_vl": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-VL-2B-Instruct/",
        "Qwen/Qwen3-VL-2B-Instruct",
    ),
}

# MoE models (slow to instantiate due to large number of experts)
MOE_MODEL_PATHS = {
    "qwen3_moe": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/",
        "Qwen/Qwen3-30B-A3B",
    ),
    "qwen3_5_moe": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-35B-A3B",
    ),
    "qwen3_vl_moe": get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-VL-30B-A3B-Instruct/",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ),
}

# Combined for backward compatibility
MODEL_PATHS = {**DENSE_MODEL_PATHS, **MOE_MODEL_PATHS}


def load_archon_model(
    model_path: str, dtype: torch.dtype = torch.bfloat16, skip_unsupported: bool = False
):
    """Load Archon model with same weights as HuggingFace model.

    Args:
        model_path: Path to the HuggingFace model.
        dtype: Data type for the model.
        skip_unsupported: If True, return None for unsupported models instead of raising.

    Returns:
        Tuple of (model, adapter) or (None, None) if skip_unsupported and model not supported.
    """
    from areal.infra.platforms import current_platform

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type

    if not is_supported_model(model_type):
        if skip_unsupported:
            return None, None
        raise ValueError(f"Model type {model_type} not supported by Archon")

    spec = get_model_spec(model_type)
    model_args = spec.model_args_class.from_hf_config(config, is_critic=False)

    with torch.device(current_platform.device_type):
        model = spec.model_class(model_args)

    # Load HF weights and convert
    hf_state_dict = get_state_dict_from_repo_id_or_path(model_path)
    adapter = spec.state_dict_adapter_class(config)
    archon_state_dict = adapter.from_hf(hf_state_dict)

    model.load_state_dict(archon_state_dict, strict=False)
    model = model.to(dtype)
    model.eval()

    return model, adapter


class TestWorkflow(RolloutWorkflow):
    """Simple test workflow for testing RolloutWorkflow functionality."""

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        await asyncio.sleep(0.1)
        prompt_len = random.randint(2, 8)
        gen_len = random.randint(2, 8)
        seqlen = prompt_len + gen_len
        return dict(
            input_ids=torch.randint(
                0,
                100,
                (
                    1,
                    seqlen,
                ),
            ),
            attention_mask=torch.ones(1, seqlen, dtype=torch.bool),
            loss_mask=torch.tensor(
                [0] * prompt_len + [1] * gen_len, dtype=torch.bool
            ).unsqueeze(0),
            rewards=torch.randn(1),
        )
