# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from areal.models.tree_attn.constants import BLOCK_SIZE, USE_TRITON_TREE_ATTN
from areal.models.tree_attn.triton_kernel import TRITON_AVAILABLE, tree_attention
from areal.utils import logging

logger = logging.getLogger("TreeAttentionFSDP")

_FLEX_DYNAMIC = not (os.environ.get("AREAL_DISABLE_FLEX_ATTENTION_DYNAMIC", "0") == "1")
# max_autotune benchmarks kernel candidates with concrete tensor sizes, which is
# incompatible with dynamic=True (symbolic shapes).  Disable it in dynamic mode.
_TORCH_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": not _FLEX_DYNAMIC,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}
logger.info(
    "Compiled torch flex attention. Options: %s, dynamic: %s",
    str(_TORCH_COMPILE_OPTIONS),
    str(_FLEX_DYNAMIC),
)
_flex_attention = torch.compile(
    flex_attention,
    dynamic=_FLEX_DYNAMIC,
    options=_TORCH_COMPILE_OPTIONS,
)
logger.info("Using block mask in flex attention, block size: %d", BLOCK_SIZE)


def create_block_mask_from_dense(
    attention_mask: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> BlockMask:
    """Create a flex attention block mask from a dense attention mask.

    This function should be called early (during data preparation) to allow
    the dense mask to be released and save memory.

    Parameters
    ----------
    attention_mask : torch.Tensor
        Dense attention mask of shape (seq_len, seq_len).
    seq_len : int
        Sequence length.
    device : torch.device
        Device to create the block mask on.

    Returns
    -------
    BlockMask
        The created block mask for use with flex_attention.
    """

    def arbitrary_mask(
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
    ):
        return attention_mask[q_idx, k_idx]

    block_mask = create_block_mask(
        arbitrary_mask,
        B=1,
        H=1,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
        _compile=False,
    )
    return block_mask


def _tree_attn_fwd_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    *args,
    **kwargs,
):
    # Check for Triton path (dict key prefixed with "tree_" to avoid
    # collisions with HuggingFace's own kwargs during **kwargs forwarding)
    tree_triton_data = kwargs.get("tree_triton_data", None)

    if USE_TRITON_TREE_ATTN and tree_triton_data is not None and TRITON_AVAILABLE:
        # [B, S, H, D] -> [B, H, S, D]
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()

        output = tree_attention(
            query,
            key,
            value,
            tree_triton_data.packed_mask,
            tree_triton_data.kv_indices,
            tree_triton_data.kv_offsets,
            tree_triton_data.q_indices,
            tree_triton_data.q_offsets,
            sm_scale=softmax_scale,
        )
        # [B, H, S, D] -> [B, S, H, D]
        output = output.permute(0, 2, 1, 3).contiguous()
        return output
    else:
        # Require pre-created block_mask
        tree_block_mask = kwargs.get("tree_block_mask", None)
        if tree_block_mask is None or not isinstance(tree_block_mask, BlockMask):
            raise ValueError(
                "_tree_attn_fwd_func requires a pre-created BlockMask in "
                "kwargs['tree_block_mask']. "
                "Use create_block_mask_from_dense() during data preparation."
            )

        # [B, S, H, D] -> [B, H, S, D]
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()

        enable_gqa = query.shape[1] != key.shape[1]

        output = _flex_attention(
            query,
            key,
            value,
            block_mask=tree_block_mask,
            score_mod=None,
            scale=softmax_scale,
            enable_gqa=enable_gqa,
        )
        # [B, H, S, D] -> [B, S, H, D]
        output = output.permute(0, 2, 1, 3).contiguous()
        return output


ORIGINAL_FLASH_ATTENTION_FORWARD = None


def patch_fsdp_for_tree_training(enable: bool = True):
    if not enable:
        return

    global ORIGINAL_FLASH_ATTENTION_FORWARD
    if ORIGINAL_FLASH_ATTENTION_FORWARD is not None:
        logger.warning("FSDP patch for tree training is already applied.")
        return

    from transformers.integrations import flash_attention

    ORIGINAL_FLASH_ATTENTION_FORWARD = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _tree_attn_fwd_func
    logger.info(
        "Patched transformers.integrations.flash_attention._flash_attention_forward "
        "with tree implementation."
    )


def restore_patch_fsdp_for_tree_training():
    """Restore original flash attention forward function. Only used in testing."""
    global ORIGINAL_FLASH_ATTENTION_FORWARD
    if ORIGINAL_FLASH_ATTENTION_FORWARD is None:
        logger.warning(
            "FSDP patch for tree training was not applied or already restored."
        )
        return

    from transformers.integrations import flash_attention

    flash_attention._flash_attention_forward = ORIGINAL_FLASH_ATTENTION_FORWARD
    ORIGINAL_FLASH_ATTENTION_FORWARD = None
    logger.info(
        "Restored transformers.integrations.flash_attention._flash_attention_forward "
        "to original implementation."
    )
