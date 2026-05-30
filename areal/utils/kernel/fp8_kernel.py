# SPDX-License-Identifier: Apache-2.0

"""Unified FP8 block-wise quantization kernel.

Compatible with SGLang/vLLM FP8 block-wise weight format.
Uses 128x128 blocks by default, e4m3fn dtype.

Triton path: high-performance GPU kernel.
PyTorch fallback: pure-PyTorch, no Triton dependency.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from areal.utils.math import ceil_div

if TYPE_CHECKING:
    pass

logger = logging.getLogger("FP8Kernel")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
FP8_MIN = -FP8_MAX

# ---------------------------------------------------------------------------
# Optional Triton
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    pass



# ---------------------------------------------------------------------------
# PyTorch fallback (always available)
# ---------------------------------------------------------------------------
def _scaled_fp8_blockwise_pytorch(
    data_hp: torch.Tensor,
    block_size: list[int] | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch block-wise FP8 quantization.

    Args:
        data_hp: BF16/FP16 weight tensor, shape (M, N).
        block_size: [block_m, block_n].

    Returns:
        (fp8_weight, scale) where scale.shape == (ceil(M/block_m), ceil(N/block_n))
        and scale = absmax / FP8_MAX.
    """
    block_size0, block_size1 = block_size[0], block_size[1]
    original_shape = data_hp.shape

    # Pad to multiples of block size
    pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
    pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1
    if pad_dim0 > 0 or pad_dim1 > 0:
        data_hp = F.pad(data_hp, (0, pad_dim1, 0, pad_dim0), mode="constant", value=0)

    max_dtype = FP8_MAX
    padded_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size0
    blk_n = data_hp.shape[1] // block_size1

    # Reshape to (blk_m, block_m, blk_n, block_n) -> permute -> flatten blocks
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3).contiguous()
    data_hp = data_hp.to(torch.float32).flatten(start_dim=2)

    # Per-block absmax
    max_abs = data_hp.abs().amax(dim=-1, keepdim=True)
    scale_fp = torch.empty_like(max_abs)
    torch.div(max_dtype, max_abs.clamp_min(1e-10), out=scale_fp)
    scale_fp = torch.where(max_abs == 0, torch.ones_like(scale_fp), scale_fp)
    scale_fp = torch.where(max_abs.isinf(), torch.ones_like(scale_fp), scale_fp)

    descale_fp = torch.reciprocal(scale_fp)
    data_hp.mul_(scale_fp)
    data_hp.clamp_(min=-max_dtype, max=max_dtype)

    fp_data = data_hp.to(FP8_DTYPE)

    # Reshape back
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
    fp_data = fp_data.permute(0, 2, 1, 3).reshape(padded_shape)

    # Crop padding
    if original_shape[0] != padded_shape[0] or original_shape[1] != padded_shape[1]:
        fp_data = fp_data[: original_shape[0], : original_shape[1]].contiguous()

    return fp_data, descale_fp.squeeze(-1)


# ---------------------------------------------------------------------------
# Triton kernel (optional, preferred)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _blockwise_cast_to_fp8_triton(
        X,
        Y,
        S,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        eps,
        fp8_min,
        fp8_max,
        BLOCK_M: tl.constexpr = 128,
        BLOCK_N: tl.constexpr = 128,
    ):
        pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
        pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = off_m < M
        mask_n = off_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        x = tl.load(
            X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
        x_s = _absmax / fp8_max
        s_inv = 1.0 / x_s
        y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

        tl.store(
            Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask
        )
        tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)

    def _blockwise_cast_to_fp8_triton_wrapper(
        x: torch.Tensor,
        block_size: list[int] | tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
        M, N = x.shape
        y = torch.empty(M, N, device=x.device, dtype=FP8_DTYPE)
        s = torch.empty(
            ceil_div(M, BLOCK_M),
            ceil_div(N, BLOCK_N),
            dtype=torch.float32,
            device=x.device,
        )

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        kwargs = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": 8 if x.is_contiguous() else 1,
            "num_stages": 2 if x.is_contiguous() else 4,
        }
        _blockwise_cast_to_fp8_triton[grid](
            x,
            y,
            s,
            *x.stride(),
            *y.stride(),
            *s.stride(),
            M,
            N,
            1e-10,
            FP8_MIN,
            FP8_MAX,
            **kwargs,
        )
        return y, s


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------
def scaled_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a 2D tensor to FP8 with block-wise quantization.

    Args:
        data_hp: Input tensor of shape (M, N). Must be 2D.
        weight_block_size: Block size as [BLOCK_M, BLOCK_N].
            Defaults to [128, 128].

    Returns:
        Tuple of (fp8_data, scale):
            - fp8_data: FP8 quantized tensor of original shape.
            - scale: Per-block scale factors of shape
              (ceil(M/BLOCK_M), ceil(N/BLOCK_N)).
              scale = absmax / FP8_MAX. Dequantize with: weight * scale.
    """
    assert len(data_hp.shape) == 2, f"Only 2D input supported, got {data_hp.shape}"

    if weight_block_size is None:
        weight_block_size = [128, 128]

    if _TRITON_AVAILABLE and os.environ.get("DISABLE_TRITON_FP8", "0") != "1":
        # Triton path with auto-padding
        block_size0, block_size1 = weight_block_size[0], weight_block_size[1]
        original_shape = data_hp.shape
        pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
        pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1

        if pad_dim0 > 0 or pad_dim1 > 0:
            data_hp = F.pad(
                data_hp, (0, pad_dim1, 0, pad_dim0), mode="constant", value=0
            )

        fp_data, scale = _blockwise_cast_to_fp8_triton_wrapper(data_hp, weight_block_size)

        if pad_dim0 > 0 or pad_dim1 > 0:
            fp_data = fp_data[: original_shape[0], : original_shape[1]].contiguous()

        return fp_data, scale

    # PyTorch fallback
    logger.debug("Triton unavailable or disabled, using PyTorch fallback for FP8 quant")
    return _scaled_fp8_blockwise_pytorch(data_hp, weight_block_size)


# ---------------------------------------------------------------------------
# Parameter filtering (which layers to quantize)
# ---------------------------------------------------------------------------
def should_quantize_param(param_name: str) -> bool:
    """Determine whether a parameter should be quantized to FP8.

    Matches SGLang's FP8 quantization rules. Only Linear weight layers
    are quantized; embeddings, norms, and output heads are skipped.
    """
    if not param_name.endswith(".weight"):
        return False

    param_lower = param_name.lower()

    # Exclude patterns
    exclude_patterns = [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "norm",
        "ln_",
        "embeddings",
        "mlp.gate.weight",  # MoE router
    ]
    for pattern in exclude_patterns:
        if pattern in param_lower:
            return False

    # Include patterns (Linear layers)
    include_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "mlp",
    ]
    for pattern in include_patterns:
        if pattern in param_lower:
            return True

    return False
