# SPDX-License-Identifier: Apache-2.0
"""把 FlashAttention-4（`flash_attn.cute`）注册成 transformers 的 attention backend。

**为什么需要这个文件**

Blackwell（B200 / sm_100）上：
  - FA2 没有 sm100 kernel，编不出来；
  - FA3 是 Hopper-only；
  - 只有 FA4（`flash-attn-4` 包，import 路径 `flash_attn.cute`）支持 sm100，
    且前向/反向都有。

但 transformers（截至 5.3.0）没有 `flash_attention_4` 这个 backend——
它的 `flash_attention_2` 路径走的是 FA2 的老 API `from flash_attn import flash_attn_func`，
而 `flash-attn-4` 装出来的 `flash_attn` 是个只含 `cute` 子模块的 namespace package，
于是报 `cannot import name 'flash_attn_func' from 'flash_attn' (unknown location)`。
参见 huggingface/transformers#44559。

本模块用 transformers 的 `AttentionInterface.register()` 补上这个 backend，
之后 `attn_implementation="flash_attention_4"` 即可正常使用。

AReaL 侧的输入是**打包过的变长序列**（batch=1，附带 `cu_seq_lens_q/k`、
`max_length_q/k`），正好对应 FA4 的 `flash_attn_varlen_func`。
"""

from typing import Any

import torch

from areal.utils import logging

logger = logging.getLogger("FA4")

ATTN_IMPL_NAME = "fa4"  # 名字不能含 "flash"：transformers 的 is_flash_attention_requested 是 `"flash" in name`，
# 一旦命中就会去 lazy-import FA2/FA3/hub-kernels，绕不过我们注册的实现。

_registered = False


def fa4_available() -> bool:
    """FA4 是否可用（装了 flash-attn-4 且能 import 到 cute 后端）。"""
    try:
        from flash_attn.cute import flash_attn_func  # noqa: F401
        from flash_attn.cute import flash_attn_varlen_func  # noqa: F401

        return True
    except Exception:
        return False


def _unwrap(out: Any) -> torch.Tensor:
    """FA4 在 return_lse=True 时返回 (out, lse)，统一取出 out。"""
    return out[0] if isinstance(out, tuple) else out


def fa4_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """transformers attention-interface 约定的签名。

    入参 q/k/v 形状是 (batch, num_heads, seq, head_dim)；
    FA4 和 FA2 一样吃 (…, seq, num_heads, head_dim)，所以先 transpose。
    """
    # ── 树训练分支 ─────────────────────────────────────────────────────
    # enable_tree_training 打开时，AReaL 会把 trie 打包后的注意力掩码作为
    # tree_block_mask / tree_triton_data 透传进来。它官方的做法是 monkey-patch
    # transformers.integrations.flash_attention._flash_attention_forward
    # （module_fsdp.patch_fsdp_for_tree_training），但那条路径只在 attn_impl 是
    # flash_attention_2 时才被调用 —— 而 B200 上 FA2 根本导入不了
    # （`cannot import name 'flash_attn_func'`），我们走的是自注册的 fa4。
    # 不在这里接管的话，树打包后的序列会被当成普通因果序列算，**分支之间会互相
    # 看见**，是静默的错误结果。
    if kwargs.get("tree_block_mask") is not None or kwargs.get("tree_triton_data") is not None:
        from areal.models.tree_attn.module_fsdp import _tree_attn_fwd_func

        out = _tree_attn_fwd_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_mask,
            scaling,
            **kwargs,
        )
        return out, None

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if dropout not in (0.0, None):
        raise ValueError("FlashAttention-4 不支持 attention dropout（dropout != 0）。")

    # (b, h, s, d) -> (b, s, h, d)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 精度：FA4 只接受 bf16/fp16。混精下 query 可能是 fp32，跟随 value 的 dtype。
    if q.dtype not in (torch.bfloat16, torch.float16):
        target = v.dtype if v.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        q, k, v = q.to(target), k.to(target), v.to(target)

    causal = module.is_causal if is_causal is None else is_causal
    window = (sliding_window - 1, 0) if sliding_window else (None, None)

    common = dict(
        softmax_scale=scaling,
        causal=causal,
        window_size=window,
        softcap=0.0 if softcap is None else float(softcap),
    )

    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")

    if cu_q is not None and cu_k is not None:
        # 打包变长路径：AReaL 的常规路径（batch 恒为 1）
        b, s = q.shape[0], q.shape[1]
        out = flash_attn_varlen_func(
            q.reshape(b * s, *q.shape[2:]),
            k.reshape(b * s, *k.shape[2:]),
            v.reshape(b * s, *v.shape[2:]),
            cu_seqlens_q=cu_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=int(kwargs.get("max_length_q", s)),
            max_seqlen_k=int(kwargs.get("max_length_k", s)),
            **common,
        )
        attn_output = _unwrap(out).reshape(b, s, *q.shape[2:])
    else:
        # 稠密路径（没打包时，比如某些 eval / VLM 分支）
        if attention_mask is not None:
            raise ValueError(
                "fa4 目前只支持打包变长输入或无 padding 的稠密输入，"
                "收到了非空 attention_mask。请改用 sdpa。"
            )
        attn_output = _unwrap(flash_attn_func(q, k, v, **common))

    return attn_output.to(value.dtype), None


def register_fa4() -> bool:
    """把 FA4 注册进 transformers 的 ALL_ATTENTION_FUNCTIONS。幂等。

    Returns
    -------
    bool
        注册成功（或此前已注册）返回 True；FA4 不可用返回 False。
    """
    global _registered
    if _registered:
        return True
    if not fa4_available():
        return False

    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_IMPL_NAME, fa4_attention_forward)
    _registered = True
    logger.info(f"Registered transformers attention backend '{ATTN_IMPL_NAME}'.")
    return True


def maybe_register_fa4(attn_impl: str) -> None:
    """建模型前调用：若用户选了 flash_attention_4，就确保它已注册。"""
    if attn_impl != ATTN_IMPL_NAME:
        return
    if not register_fa4():
        raise RuntimeError(
            "attn_impl='fa4' 但 FA4 不可用。"
            "请安装 flash-attn-4（`from flash_attn.cute import flash_attn_func` 需能 import）。"
        )
