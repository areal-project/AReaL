# SPDX-License-Identifier: Apache-2.0

"""Fast MindSpeed triton ``l2norm`` for GDN (Qwen3.5/3.6) on NPU.

Bundles fla PR #42 (``do_not_specialize`` + no ``@triton.autotune``) so l2norm --
the only triton kernel in the GDN forward -- compiles ONCE instead of re-autotuning
per shape, the dominant cost of the GDN first-step compile. Monkeypatches
``mindspeed.ops.triton.l2norm`` in place, so it MUST be imported BEFORE
``mindspeed.megatron_adaptor`` (which binds ``l2norm`` into ``fla.modules``). No-op
without MindSpeed; tracked in-repo so it survives image rebuilds.

Also despecializes ``NT`` in the GDN chunk kernels (``_despecialize_chunk_nt``) so
they compile once across sequence lengths instead of recompiling per seqlen -- the
dominant per-new-seqlen chunk cost. All three are runtime-free compile-once wins.
"""
# Copyright c) 2023-2025 Songlin Yang Yu Zhang
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from mindspeed.ops.triton.utils import input_guard

BT_LIST = (8, 16, 32, 64, 128)
FWD_TARGET_BLOCK_ELEMENTS = 8 * 1024
BWD_TARGET_BLOCK_ELEMENTS = 8 * 1024
FWD_MAX_BT = 32
BWD_MAX_BT = 64


def _select_l2norm_bt(bd: int, target_block_elements: int, max_bt: int) -> int:
    max_bt_by_size = max(1, target_block_elements // max(1, int(bd)))
    max_bt = min(int(max_bt), max_bt_by_size)
    candidates = [bt for bt in BT_LIST if bt <= max_bt]
    return candidates[-1] if candidates else BT_LIST[0]


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t = tl.program_id(0)
    for offset in range(0, bt_size):
        block_start = (i_t * bt_size + offset) * BT
        if block_start < T:
            p_x = tl.make_block_ptr(
                x, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0)
            )
            p_y = tl.make_block_ptr(
                y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0)
            )
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))

            b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
            b_y = b_x * b_rstd[:, None]

            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t = tl.program_id(0)
    for offset in range(0, bt_size):
        block_start = (i_t * bt_size + offset) * BT
        if block_start < T:
            p_y = tl.make_block_ptr(
                y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0)
            )
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))
            p_dy = tl.make_block_ptr(
                dy, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0)
            )
            p_dx = tl.make_block_ptr(
                dx, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0)
            )

            b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            b_dx = (
                b_dy * b_rstd[:, None]
                - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
            )
            tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        BT = _select_l2norm_bt(BD, FWD_TARGET_BLOCK_ELEMENTS, FWD_MAX_BT)
        bt_size = 32

        def grid(meta):
            new_bt = meta["BT"] * bt_size
            return (triton.cdiv(T, new_bt),)

        l2norm_fwd_kernel[grid](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            bt_size=bt_size,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        BT = _select_l2norm_bt(BD, BWD_TARGET_BLOCK_ELEMENTS, BWD_MAX_BT)
        bt_size = 40

        def grid(meta):
            new_bt = meta["BT"] * bt_size
            return (triton.cdiv(T, new_bt),)

        l2norm_bwd_kernel[grid](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            bt_size=bt_size,
        )
    else:
        l2norm_bwd_kernel1[(T,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            D=D,
            BD=BD,
        )

    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        y, rstd = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)


def _reduce_chunk_autotune() -> bool:
    """Collapse the GDN chunk kernels' autotune to a single config.

    Wraps ``mindspeed.ops.triton.utils.get_autotune_config`` to return only the
    first config, skipping the on-device do_bench over the rest (the bulk of the
    first-step compile). Must run before the chunk modules import. Idempotent;
    trade-off: the kept config may be sub-optimal at runtime.
    """
    try:
        import mindspeed.ops.triton.utils as _u
    except Exception:
        return False
    if getattr(_u, "_areal_autotune_reduced", False):
        return True
    _orig = getattr(_u, "get_autotune_config", None)
    if _orig is None:
        return False

    def _single_config(*args, **kwargs):
        cfgs = _orig(*args, **kwargs)
        try:
            return cfgs[:1] if cfgs else cfgs
        except TypeError:
            return cfgs

    _u.get_autotune_config = _single_config
    _u._areal_autotune_reduced = True
    return True


# GDN chunk kernels that take ``NT`` (= cdiv(T, BT), number of chunks) as a
# ``tl.constexpr``: since ``NT`` changes with sequence length, they recompile per
# seqlen (the dominant chunk first-step cost after the l2norm swap). ``T`` is
# already ``do_not_specialize``; these just miss ``NT``.
_CHUNK_NT_KERNELS = (
    (
        "mindspeed.ops.triton.chunk_delta_h",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    ),
    ("mindspeed.ops.triton.wy_fast", "recompute_w_u_fwd_kernel"),
    ("mindspeed.ops.triton.wy_fast", "prepare_wy_repr_bwd_kernel"),
)


def _despecialize_chunk_nt() -> bool:
    """Make ``NT`` a runtime arg in the GDN chunk kernels so they compile once.

    Reuses MindSpeed's own kernel body: walks the ``Heuristics[/Autotuner]/
    JITFunction`` wrapper chain to the ``JITFunction``, drops ``NT`` from the
    python fn's ``__annotations__``, re-``jit``s it with ``do_not_specialize=
    ['T','NT']``, and splices the new jit back under the original wrapper (so the
    heuristics/autotune config is preserved exactly). Must run before the chunk
    modules are first used. Idempotent; no-op without MindSpeed. Runtime-free
    (the ``for i_t in range(NT)`` loop goes static-unroll -> dynamic, which the
    varlen branch already exercises; measured warm runtime unchanged).
    """
    import importlib

    import triton

    patched = False
    for mod_name, kern_name in _CHUNK_NT_KERNELS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        wrapper = getattr(mod, kern_name, None)
        if wrapper is None:
            continue
        parent, node = None, wrapper
        while node is not None and type(node).__name__ != "JITFunction":
            parent, node = node, getattr(node, "fn", None)
        if node is None:
            continue
        pyfn = node.fn
        ann = getattr(pyfn, "__annotations__", {})
        if "NT" not in ann:
            patched = True  # already despecialized
            continue
        del ann["NT"]
        new_jit = triton.jit(do_not_specialize=["T", "NT"])(pyfn)
        if parent is None:
            setattr(mod, kern_name, new_jit)
        else:
            parent.fn = new_jit
        patched = True
    return patched


def apply() -> bool:
    """Swap in the fast l2norm + collapse chunk autotune to one config.

    Idempotent; no-op (False) without MindSpeed or if
    ``AREAL_DISABLE_TRITON_GDN_COMPILE_PATCH=1``. Must run before
    ``mindspeed.megatron_adaptor`` binds l2norm into ``fla.modules``.
    """
    if os.environ.get("AREAL_DISABLE_TRITON_GDN_COMPILE_PATCH") == "1":
        return False
    try:
        import mindspeed.ops.triton.l2norm as _ms
    except Exception:
        return False
    if not getattr(_ms, "_areal_l2norm_patched", False):
        _ms.l2norm = l2norm
        _ms.l2norm_fwd = l2norm_fwd
        _ms.l2norm_bwd = l2norm_bwd
        _ms.l2norm_fwd_kernel = l2norm_fwd_kernel
        _ms.l2norm_bwd_kernel = l2norm_bwd_kernel
        _ms.l2norm_fwd_kernel1 = l2norm_fwd_kernel1
        _ms.l2norm_bwd_kernel1 = l2norm_bwd_kernel1
        _ms._select_l2norm_bt = _select_l2norm_bt
        _ms._areal_l2norm_patched = True
    # The chunk autotune reduction has a (small) runtime-perf trade-off, so it
    # can be skipped independently while keeping the safe l2norm swap.
    if os.environ.get("AREAL_DISABLE_TRITON_CHUNK_AUTOTUNE") != "1":
        _reduce_chunk_autotune()
    # Despecialize NT in the GDN chunk kernels (compile-once across seqlens).
    # Runtime-free; separate flag for A/B.
    if os.environ.get("AREAL_DISABLE_TRITON_CHUNK_NT") != "1":
        _despecialize_chunk_nt()
    return True


# Apply on import. megatron_engine.py imports this module before MindSpeed.
_APPLIED = apply()
