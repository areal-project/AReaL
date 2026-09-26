# SPDX-License-Identifier: Apache-2.0
"""Causal PLE chunking for the Qwen 256K recipe."""

import functools

import torch


def chunked_ple(
    base,
    hc_state,
    key,
    value,
    norm_key_w,
    norm_query_w,
    norm_conv_w,
    conv1d_weight,
    n,
    eps,
    dilation,
    seq_len,
    *,
    chunk_tokens=8192,
):
    total = hc_state.shape[0]
    if chunk_tokens <= 0 or seq_len <= chunk_tokens:
        return base(
            hc_state,
            key,
            value,
            norm_key_w,
            norm_query_w,
            norm_conv_w,
            conv1d_weight,
            n,
            eps,
            dilation,
            seq_len,
        )
    if seq_len <= 0 or total % seq_len:
        raise ValueError("PLE chunking requires complete uniform sequence rows")
    halo = (conv1d_weight.shape[-1] - 1) * dilation
    # Accumulate shared parameter gradients across chunks in FP32, then cast once.
    norm_key_w, norm_query_w, norm_conv_w, conv1d_weight = (
        tensor.float()
        for tensor in (norm_key_w, norm_query_w, norm_conv_w, conv1d_weight)
    )
    pieces = []
    for row in range(0, total, seq_len):
        for start in range(0, seq_len, chunk_tokens):
            end = min(seq_len, start + chunk_tokens)
            left = max(0, start - halo)
            sl = slice(row + left, row + end)
            result = base(
                hc_state[sl],
                key[sl],
                value[sl],
                norm_key_w,
                norm_query_w,
                norm_conv_w,
                conv1d_weight,
                n,
                eps,
                dilation,
                end - left,
            )
            if result is None:
                return None
            pieces.append(result[start - left :])
    return torch.cat(pieces, dim=0)


def install(chunk_tokens=8192):
    from mcore_bridge.model.modules import ple

    if chunk_tokens <= 0:
        raise ValueError("PLE chunk_tokens must be positive")
    base = ple.ple_gate_conv_triton
    base = getattr(base, "_qwen_ple_original", base)
    wrapper = functools.partial(chunked_ple, base, chunk_tokens=chunk_tokens)
    wrapper._qwen_ple_original = base
    ple.ple_gate_conv_triton = wrapper
