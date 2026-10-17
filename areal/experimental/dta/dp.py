# SPDX-License-Identifier: Apache-2.0

# This code is adapted with minor modifications from
# https://github.com/Whisper-6/DynamicTreeAttn/blob/main/data_parallel.py.
# Special thanks to Yuchen Yang for significant contributions to the load-balanced data parallel partitioning algorithm.
from types import SimpleNamespace

from areal.experimental.dta.token_trie import TokenTrie
from areal.experimental.dta.trie import CompressedTrie, _get_stats, _get_subtrie


def LB_by_n_tokens(token_seqs, K):
    bins = [[] for _ in range(K)]
    bin_lens = [0] * K
    seq_indices = sorted(range(len(token_seqs)), key=lambda i: -len(token_seqs[i]))
    for i in seq_indices:
        min_bin = min(range(K), key=lambda j: bin_lens[j])
        bins[min_bin].append(i)
        bin_lens[min_bin] += len(token_seqs[i])
    return bins


def pred_time(
    compressed_trie, time_model, mode: str, block_size: int | None = None
) -> float:
    if mode == "forward":
        _, lens, lcp_lens = compressed_trie.get_order_forward()
    elif mode == "backward":
        _, lens, lcp_lens = compressed_trie.get_order_backward()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    stats = _get_stats(lens, lcp_lens, mode, block_size)
    return time_model.pred(stats)


def get_original_bins(
    token_trie: TokenTrie, leaf_bins: list[list[int]]
) -> list[list[int]]:
    bins = [[] for _ in range(len(leaf_bins))]
    for bucket_idx, leaf_bucket in enumerate(leaf_bins):
        for leaf_idx in leaf_bucket:
            attach_lists = token_trie.attach_lists[leaf_idx]
            for attach, _ in attach_lists:
                original_seq_idx = attach["_sequence_batch_id"]
                bins[bucket_idx].append(original_seq_idx)
    return bins


def LB_by_TM(token_seqs, time_model, config: SimpleNamespace):
    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)

    K = config.K
    leaf_bins = [[] for _ in range(K)]
    bin_times = [0.0] * K

    for i in range(n_leaf_seqs):
        min_bin = min(range(K), key=lambda j: bin_times[j])
        leaf_bins[min_bin].append(i)
        bin_compressed_trie = _get_subtrie(compressed_trie, leaf_bins[min_bin])
        bin_times[min_bin] = pred_time(
            bin_compressed_trie, time_model, config.mode, config.block_size
        )

    bins = get_original_bins(token_trie, leaf_bins)
    return bins


def try_divide(
    compressed_trie,
    n_seqs,
    config: SimpleNamespace,
    divL,
    divR,
    time_model,
    cost_limit: float,
) -> list[list[int]] | None:
    K = config.K
    divs = []

    start = 0
    while start < n_seqs:
        divs.append(start)
        if len(divs) > K:
            break
        L = max(divL[len(divs)] - 1, start)
        R = divR[len(divs)] - 1
        while L < R:
            mid = (L + R + 1) // 2
            cur_subtrie = _get_subtrie(compressed_trie, set(range(start, mid + 1)))
            est_time = pred_time(
                cur_subtrie, time_model, config.mode, config.block_size
            )
            if est_time <= cost_limit:
                L = mid
            else:
                R = mid - 1
        start = L + 1

    return divs


def LB_by_DFS_and_TM(token_seqs, time_model, config: SimpleNamespace):
    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    K = config.K
    if n_leaf_seqs == 0:
        return [[] for _ in range(K)]

    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)

    R = float(pred_time(compressed_trie, time_model, config.mode, config.block_size))
    L = R / K
    eps = R * 1e-4

    divL = [0] * (K + 1)
    # Maintain a valid initial partition boundary so K==1 (L==R) does not
    # skip the search and accidentally produce empty bins.
    divR = [0] + [n_leaf_seqs] * K

    while R - L > eps:
        mid = (L + R) / 2.0
        divs = try_divide(
            compressed_trie, n_leaf_seqs, config, divL, divR, time_model, mid
        )
        if len(divs) <= K:
            R = mid
            divR[: len(divs)] = divs
        else:
            L = mid + eps
            divL = divs[: K + 1]

    leaf_bins = [list(range(divR[i], divR[i + 1])) for i in range(K)]
    bins = get_original_bins(token_trie, leaf_bins)
    return bins


# -------- Test --------


def eval(token_seqs, bins, time_model, config: SimpleNamespace):
    total_time = 0.0
    max_time = 0.0
    for bucket in bins:
        token_trie = TokenTrie([token_seqs[i] for i in bucket])
        compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)
        bucket_pred_time = pred_time(
            compressed_trie, time_model, config.mode, config.block_size
        )
        total_time += bucket_pred_time
        max_time = max(max_time, bucket_pred_time)
    return total_time, max_time
