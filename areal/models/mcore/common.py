# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.utils import logging

logger = logging.getLogger("MCoreCommon")


T = TypeVar("T", bound=TransformerConfig)


# Modified from verl:
# https://github.com/volcengine/verl/blob/ea885f32f04d86c3a81de18083db7eef0d781421/verl/models/mcore/config_converter.py
def hf_to_mcore_base_args(
    hf_config: PretrainedConfig,
    dtype: torch.dtype,
    **override_transformer_config_kwargs,
) -> dict:
    # TODO: add parallel configs for transformer configs
    # Common parallel state parameters
    # overlap_p2p_comm = (
    #     mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    #     and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    # )
    # batch_p2p_comm = False

    # Base configuration with common parameters
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "add_bias_linear": True,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": 1,  # mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": 1,  # mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": 1,  # mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": 1,  # mpu.get_experget_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": 1,  # mpu.get_context_part_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": 1,
        "overlap_p2p_comm": False,
        "batch_p2p_comm": False,
        "sequence_parallel": False,  # mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    # override_transformer_config_kwargs as kwargs shall never be none
    base_config.update(override_transformer_config_kwargs)
    return base_config


# Modified from verl:
# https://github.com/volcengine/verl/blob/ea885f32f04d86c3a81de18083db7eef0d781421/verl/models/mcore/config_converter.py
def check_and_construct_configs(original_config: dict, cls: type[T]) -> T:
    """
    Check and disable incompatible configurations for older Megatron version.

    Args:
        original_config (dict): The original model configuration.

    Returns:
        dict: The updated model configuration with incompatible settings disabled.
    """
    removed_keys = []
    for key in original_config.keys():
        if not hasattr(cls, key):
            removed_keys.append(key)
    if removed_keys:
        logger.warning(
            f"The following keys are not supported in the current Megatron version and will be removed: {removed_keys}"
        )
        for key in removed_keys:
            original_config.pop(key)

    print(f"Overridden {cls.__name__} init config: {original_config}")
    return cls(**original_config)
