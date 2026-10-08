# SPDX-License-Identifier: Apache-2.0

"""Causal depthwise convolution over compact rollout-tree tokens."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from areal.models.tree_attn.tree import TrieNode, trie_to_parent_array


@torch.compile(dynamic=True, options={"max_autotune": False})
def _compiled_silu_tree_conv(
    projected: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    ancestry: torch.Tensor,
) -> torch.Tensor:
    """Fuse indexed ancestor reads, depthwise convolution, and SiLU."""
    count, width = projected.shape
    kernel_size = ancestry.shape[1]
    context = projected.index_select(0, ancestry.clamp_min(0).reshape(-1))
    context = context.view(count, kernel_size, width)
    context = context * (ancestry >= 0).unsqueeze(-1)
    output = (context.float() * weight[:, 0, :].t().float()).sum(dim=1)
    if bias is not None:
        output = output + bias.float()
    return F.silu(output).to(projected.dtype)


def _tree_conv_ancestry(
    trie: TrieNode, count: int, kernel_size: int, device: torch.device
) -> torch.Tensor:
    """Reuse the immutable ancestor window across all GDN layers of a tree."""
    cache = getattr(trie, "_causal_conv_ancestry_cache", None)
    if cache is None:
        cache = {}
        trie._causal_conv_ancestry_cache = cache
    key = (count, kernel_size, device)
    if key not in cache:
        parent = trie_to_parent_array(trie, count).squeeze(0).long()
        indices = torch.arange(count)
        window = []
        for _ in range(kernel_size):
            window.append(indices)
            indices = torch.where(indices >= 0, parent[indices.clamp_min(0)], -1)
        cache[key] = torch.stack(window[::-1], dim=1).to(device)
    return cache[key]


def tree_causal_conv1d(
    projected: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    trie: TrieNode,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Convolve each compact token with its true ancestor window once.

    ``projected`` is ``[C, D]`` and ``weight`` is the depthwise convolution
    weight ``[D, 1, K]``. Padding before each root is zero.
    """
    count, width = projected.shape
    kernel_size = weight.shape[-1]
    ancestry = _tree_conv_ancestry(trie, count, kernel_size, projected.device)
    if projected.is_cuda and getattr(activation, "__name__", None) in {
        "silu",
        "swish",
    }:
        return _compiled_silu_tree_conv(projected, weight, bias, ancestry)
    context = projected.index_select(0, ancestry.clamp_min(0).reshape(-1))
    context = context.view(count, kernel_size, width)
    context = context * (ancestry >= 0).unsqueeze(-1)
    output = (context.float() * weight[:, 0, :].t().float()).sum(dim=1)
    if bias is not None:
        output = output + bias.float()
    return activation(output).to(projected.dtype)
