# SPDX-License-Identifier: Apache-2.0

# FP8 Triton kernels for quantization and dequantization
# Adapted from slime

import torch
import triton
import triton.language as tl

from areal.utils.math import ceil_div


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
    BLOCK_M: tl.constexpr = 32,
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


def blockwise_cast_to_fp8_triton(
    x: torch.Tensor, block_size: list[int] | tuple[int, int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise FP8 quantization using Triton.

    Args:
        x: Input tensor (2D)
        block_size: Optional [block_m, block_n], defaults to [128, 128]

    Returns:
        (quantized_weight, scale)
    """
    BLOCK_M, BLOCK_N = 128, 128
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    M, N = x.shape
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = -fp8_max
    y = torch.empty(M, N, device=x.device, dtype=fp8_dtype)
    s = torch.empty(
        ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device
    )

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if x.is_contiguous():
        kwargs = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": 8,
            "num_stages": 2,
        }
    else:
        kwargs = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": 1,
            "num_stages": 4,
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
        fp8_min,
        fp8_max,
        **kwargs,
    )
    return y, s


# Adapted from https://github.com/alibaba/Pai-Megatron-Patch/blob/2b201af08336dea0403df7c6b497c964cf5a2e75/toolkits/model_checkpoints_convertor/deepseek/fp8_cast_bf16.py
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dst_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights to the given dtype.

    Args:
        x: FP8 weight tensor (2D)
        s: Scale inverse tensor (2D, shape matches block structure)
        block_size: Block size used for quantization
        dst_dtype: Destination dtype to dequantize to

    Returns:
        Dequantized weight tensor in the destination dtype
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dst_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y
