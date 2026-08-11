# SPDX-License-Identifier: Apache-2.0

from transformers.integrations.hub_kernels import is_kernel

BUILTIN_ATTN_IMPLS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    # FA4（flash_attn.cute）。transformers 没有这个 backend，由 fa4_attn.py 注册；
    # 名字不能含 "flash"（否则被 transformers 抢走）。Blackwell(sm100) 上唯一可用的 FA。
    "fa4",
    "flex_attention",
)


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
