# SPDX-License-Identifier: Apache-2.0

# The following code is adapted with minor modifications from
# https://github.com/Whisper-6/DynamicTreeAttn/blob/main/token_trie.py.
# Special thanks to Yuchen Yang for outstanding contributions to the optimized DFS order.

import torch

from areal.experimental.dta.trie import CompressedTrie, _get_stats


def _lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Compute the length of the longest common prefix of two 1D tensors."""
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())


def _leafization(input_ids: list[torch.LongTensor], attachs: list[dict]):
    """
    Args:
        input_ids: List of token tensors, sorted in lexicographic order.
        attachs: List of dicts, each storing loss-related config for one token tensor.

    Merge fully overlapping prefixes and compute the `lcp_lens` list.
    """

    # Compute adjacent LCP lengths and validate lexicographic ordering.
    lcp_lens = []
    for i in range(len(input_ids) - 1):
        seq_L, seq_R = input_ids[i], input_ids[i + 1]
        lcp = _lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("input_ids not sorted in lexicographic order.")
        lcp_lens.append(lcp)

    # Merge fully overlapping prefixes by keeping only the longest sequence.
    input_ids_leafed = []
    attach_lists = []
    lcp_lens_leafed = []

    fork = -1
    for i in range(len(input_ids)):
        if i == len(input_ids) - 1 or lcp_lens[i] < min(
            input_ids[i].numel(), input_ids[i + 1].numel()
        ):
            input_ids_leafed.append(input_ids[i])
            if i < len(input_ids) - 1:
                lcp_lens_leafed.append(lcp_lens[i])
            attach_list = []
            for k in range(fork + 1, i + 1):
                attach_list.append((attachs[k], input_ids[k].numel()))
            attach_lists.append(attach_list)
            fork = i

    return input_ids_leafed, attach_lists, lcp_lens_leafed


class TokenTrie:
    def __init__(
        self,
        inputs: list[torch.LongTensor],
        attachs: list[dict] | None = None,
        sorted: bool = False,
    ):
        if attachs is not None:
            if len(inputs) != len(attachs):
                raise ValueError("Length of inputs and attachs must match.")
        else:
            attachs = [{} for _ in range(len(inputs))]

        # Attach the original sequence index to each attachment dict.
        for seq_id in range(len(inputs)):
            attachs[seq_id]["_sequence_batch_id"] = seq_id

        # -------- sort by lexicographical order of input_ids --------
        if not sorted:
            pairs = list(zip(inputs, attachs))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted, attachs_sorted = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            inputs_sorted, attachs_sorted = inputs, attachs

        # -------- leafization --------
        self.inputs, self.attach_lists, self.lcp_lens = _leafization(
            inputs_sorted, attachs_sorted
        )
        self.lens = [len(ids) for ids in self.inputs]

        # -------- stats --------
        self.n_sequences = len(inputs)
        self.n_tokens = sum(len(ids) for ids in inputs)

    def get_stats(self, mode: str, block_size: int | None = None):
        stats = _get_stats(self.lens, self.lcp_lens, mode, block_size)
        stats["n_sequences"] = self.n_sequences
        stats["n_tokens"] = self.n_tokens
        return stats

    def permute(self, order):
        self.inputs = [self.inputs[i] for i in order]
        self.attach_lists = [self.attach_lists[i] for i in order]
        self.lens = [self.lens[i] for i in order]
        self.lcp_lens = [
            _lcp_torch(self.inputs[i], self.inputs[i + 1])
            for i in range(len(self.inputs) - 1)
        ]

    def forward_permute(self):
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order, _, _ = compressed_trie.get_order_forward()
        self.permute(order)

    def backward_permute(self):
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order, _, _ = compressed_trie.get_order_backward()
        self.permute(order)

    def random_permute(self):
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order = compressed_trie.get_order_random()
        self.permute(order)
