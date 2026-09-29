# SPDX-License-Identifier: Apache-2.0

"""Causal depthwise convolution over compact rollout-tree tokens."""

from collections.abc import Callable

import torch

from areal.models.tree_attn.tree import TrieNode, trie_to_parent_array


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
    parent = trie_to_parent_array(trie, count).squeeze(0).long()
    indices = torch.arange(count)
    window = []
    for _ in range(kernel_size):
        window.append(indices)
        indices = torch.where(indices >= 0, parent[indices.clamp_min(0)], -1)
    ancestry = torch.stack(window[::-1], dim=1).to(projected.device)
    context = projected.index_select(0, ancestry.clamp_min(0).reshape(-1))
    context = context.view(count, kernel_size, width)
    context = context * (ancestry >= 0).unsqueeze(-1)
    output = (context.float() * weight[:, 0, :].t().float()).sum(dim=1)
    if bias is not None:
        output = output + bias.float()
    return activation(output).to(projected.dtype)
