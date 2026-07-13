# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import torch
from mbridge.core import LLMBridge
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch.nn.attention.flex_attention import BlockMask

from areal.models.tree_attn.constants import USE_TRITON_TREE_ATTN
from areal.models.tree_attn.module_fsdp import (
    _flex_attention,
    create_block_mask_from_dense,
)
from areal.models.tree_attn.triton_kernel import (
    TRITON_AVAILABLE,
    TreeAttentionData,
    tree_attention,
)
from areal.utils import logging

logger = logging.getLogger("TreeAttentionMegatron")


class PytorchFlexAttention(torch.nn.Module):
    """Pytorch flex attention implementation that supports arbitrary attention mask type."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale
        logger.info("Using PytorchFlexAttention for tree training attention")

        # PytorchFlexAttention does not support context parallel
        if config.context_parallel_size != 1:
            raise ValueError(
                "PytorchFlexAttention does not support context parallelism."
            )

        if attention_type != "self":
            raise ValueError("PytorchFlexAttention only supports self-attention.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: BlockMask | TreeAttentionData | torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # query: [S, B, H, D] in which B should be 1 in current tree training implementation
        # key: [S, B, H, D]
        # value: [S, B, H, D]
        # attention_mask: BlockMask | TreeAttentionData | torch.Tensor
        #   - BlockMask: pre-created block mask for flex_attention
        #   - TreeAttentionData: for Triton tree attention
        #   - torch.Tensor: dense attention mask that will be converted to BlockMask
        # attention_mask_type: arbitrary

        # Check for Triton path
        if (
            USE_TRITON_TREE_ATTN
            and isinstance(attention_mask, TreeAttentionData)
            and TRITON_AVAILABLE
        ):
            # Triton path
            if attention_bias is not None:
                raise NotImplementedError("Triton path does not support attention_bias")
            if packed_seq_params is not None:
                raise NotImplementedError(
                    "Triton path does not support packed sequences"
                )

            # Input: [S, B, H, D], Triton expects [B, H, N, D]
            query = query.permute(1, 2, 0, 3).contiguous()
            key = key.permute(1, 2, 0, 3).contiguous()
            value = value.permute(1, 2, 0, 3).contiguous()

            output = tree_attention(
                query,
                key,
                value,
                attention_mask.packed_mask,
                attention_mask.kv_indices,
                attention_mask.kv_offsets,
                attention_mask.q_indices,
                attention_mask.q_offsets,
                sm_scale=self.softmax_scale,
            )
            # Output: [B, H, S, D], convert to [S, B, H*D]
            output = output.permute(2, 0, 1, 3).contiguous()
            output = output.view(output.shape[0], output.shape[1], -1)
            return output
        else:
            # Existing flex_attention path
            if attention_bias is not None:
                raise NotImplementedError(
                    "PytorchFlexAttention does not support attention_bias yet."
                )
            if packed_seq_params is not None:
                raise NotImplementedError(
                    "PytorchFlexAttention does not support packed sequences yet."
                )

            if isinstance(attention_mask, torch.Tensor):
                seq_len = attention_mask.shape[0]
                block_mask = create_block_mask_from_dense(
                    attention_mask, seq_len, attention_mask.device
                )
            elif isinstance(attention_mask, BlockMask):
                block_mask = attention_mask
            else:
                raise ValueError(
                    "PytorchFlexAttention requires either a BlockMask, dense tensor, "
                    "or TreeAttentionData as attention_mask. "
                    "For flex_attention path, use create_block_mask_from_dense() "
                    "or pass a dense mask tensor."
                )

            # query, key, value shape: [S, B, H, D] -> [B, H, S, D]
            query = query.permute(1, 2, 0, 3)
            key = key.permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
            enable_gqa = query.shape[1] != key.shape[1]

            output = _flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=None,
                scale=self.softmax_scale,
                enable_gqa=enable_gqa,
            )

            # output shape: [B, H, S, D] -> [S, B, H, D] -> [S, B, H*D]
            output = (
                output.permute(2, 0, 1, 3)
                .contiguous()
                .view(output.shape[2], output.shape[0], -1)
            )
            return output


@contextmanager
def patch_bridge_for_tree_training(enable: bool = True):
    """Context manager to patch LLMBridge for tree training with arbitrary attention mask.

    Parameters
    ----------
    enable : bool, default=True
        If True, apply the patch. If False, the context manager is a no-op.

    Yields
    ------
    None

    Examples
    --------
    >>> with patch_bridge_for_tree_training(enable=True):
    ...     # LLMBridge is patched here
    ...     model = create_model()
    ... # Patch is reverted after exiting the context
    """
    if not enable:
        yield
        return

    # Store original method
    original_layer_spec_getter = LLMBridge._get_transformer_layer_spec

    def _patched_getter(self, vp_stage: int | None = None):
        spec: TransformerBlockSubmodules = original_layer_spec_getter(self, vp_stage)
        for layer_spec in spec.layer_specs:
            if layer_spec.module is not TransformerLayer:
                logger.info(f"Skipping patch module: {layer_spec.module}")
                continue
            submodules: TransformerLayerSubmodules = layer_spec.submodules
            self_attn_spec = submodules.self_attention
            if self_attn_spec.module is not SelfAttention:
                logger.info(f"Skipping patch module: {self_attn_spec.module}")
                continue
            self_attn_spec.params["attn_mask_type"] = AttnMaskType.arbitrary
            self_attn_spec.submodules.core_attention = PytorchFlexAttention
        return spec

    # Apply patch
    LLMBridge._get_transformer_layer_spec = _patched_getter
    try:
        yield
    finally:
        # Revert patch
        LLMBridge._get_transformer_layer_spec = original_layer_spec_getter
