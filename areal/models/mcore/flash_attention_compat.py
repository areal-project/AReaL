# SPDX-License-Identifier: Apache-2.0

import inspect


def patch_te_flash_attention_3() -> bool:
    """Adapt TE's CP calls to FA3 packages exposing the older tuple-window API.

    Only TE's low-level aliases change; the public FA3 API and CUDA kernels are
    untouched. Newer FA3 interfaces and repeated calls need no adaptation.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    forward = getattr(backends, "_flash_attn_fwd_v3", None)
    if forward is None or "window_size_left" in inspect.signature(forward).parameters:
        return False
    backward = backends._flash_attn_bwd_v3

    def forward_with_window_sides(
        *args, window_size_left=-1, window_size_right=-1, **kwargs
    ):
        return forward(
            *args, window_size=(window_size_left, window_size_right), **kwargs
        )

    def backward_with_window_sides(
        *args, is_causal=False, window_size_left=-1, window_size_right=-1, **kwargs
    ):
        return backward(
            *args,
            causal=is_causal,
            window_size=(window_size_left, window_size_right),
            **kwargs,
        )[3]

    backends._flash_attn_fwd_v3 = forward_with_window_sides
    backends._flash_attn_bwd_v3 = backward_with_window_sides
    return True
