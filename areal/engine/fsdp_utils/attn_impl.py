# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AttentionInterface
from transformers.integrations.hub_kernels import is_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward

BUILTIN_ATTN_IMPLS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
)

# Backends that never read ``cu_seq_lens_q/k``. With packed micro-batches and no
# mask they apply one causal mask over the whole row, so each sequence attends to
# the sequences packed before it. sdpa is split per sequence by
# ``packed_sdpa_attention_forward``; eager gets a block-diagonal mask.
PACKED_MASK_ATTN_IMPLS = ("sdpa", "eager")

# Model kwarg carrying packed sequence boundaries as a tuple of Python ints, so the
# attention function never has to sync the device to read ``cu_seqlens``.
PACKED_SEQ_BOUNDS_KWARG = "packed_seq_bounds"


def is_valid_attn_impl(attn_impl: str) -> bool:
    """Return whether ``attn_impl`` is a builtin backend or valid HF kernels syntax."""
    return attn_impl in BUILTIN_ATTN_IMPLS or is_kernel(attn_impl)


def get_attn_impl_validation_error(attn_impl: str) -> str:
    """Build the canonical validation message for invalid attention implementations."""
    return (
        "attn_impl must be one of "
        f"{BUILTIN_ATTN_IMPLS} or a Hugging Face kernels repo ID "
        "formatted as org/repo[@revision][:entrypoint] "
        "(optionally prefixed with wrapper|), "
        f"got '{attn_impl}'."
    )


def build_packed_attention_mask(
    cu_seqlens: torch.Tensor,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_impl: str,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build a ``[1, 1, seq_len, seq_len]`` causal mask that keeps packed sequences apart.

    Positions past ``cu_seqlens[-1]`` (trailing padding) form their own segment.
    sdpa takes a boolean mask (True = attend); eager adds the mask to the attention
    scores, so it gets an additive mask with ``finfo(dtype).min`` on blocked entries.
    """
    pos = torch.arange(seq_len, device=device)
    seg = torch.bucketize(
        pos, cu_seqlens[1:].to(device=device, dtype=pos.dtype), right=True
    )
    allowed = (seg[:, None] == seg[None, :]) & (pos[:, None] >= pos[None, :])
    if sliding_window is not None:
        allowed &= (pos[:, None] - pos[None, :]) < sliding_window
    allowed = allowed[None, None]
    if attn_impl == "sdpa":
        return allowed
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


def packed_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """sdpa that runs each packed sequence separately when boundaries are given.

    A full block-diagonal mask would force sdpa onto its math kernel, which stores
    an ``[heads, seq_len, seq_len]`` score matrix per layer. Slicing per sequence
    keeps the fused causal kernel and its memory footprint. Without the
    ``packed_seq_bounds`` kwarg this is exactly transformers' ``sdpa``.
    """
    bounds = kwargs.pop(PACKED_SEQ_BOUNDS_KWARG, None)
    if bounds is None or attention_mask is not None or query.shape[0] != 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    seq_len = query.shape[2]
    if bounds[-1] < seq_len:  # trailing padding forms its own segment
        bounds = (*bounds, seq_len)
    sliding_window = getattr(module, "sliding_window", None)
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask = None
        if sliding_window is not None and end - start > sliding_window:
            mask = build_packed_attention_mask(
                torch.tensor([0, end - start]),
                end - start,
                device=query.device,
                dtype=query.dtype,
                attn_impl="sdpa",
                sliding_window=sliding_window,
            )
        out, _ = sdpa_attention_forward(
            module,
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            mask,
            **kwargs,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), None


def register_packed_sdpa() -> None:
    """Route transformers' ``sdpa`` through ``packed_sdpa_attention_forward``.

    Idempotent. Calls that do not pass ``packed_seq_bounds`` are unaffected.
    """
    AttentionInterface.register("sdpa", packed_sdpa_attention_forward)
