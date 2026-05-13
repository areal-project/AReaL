from types import SimpleNamespace

import torch

from areal.experimental.dta.dp import (
    LB_by_DFS_and_TM,
    LB_by_n_tokens,
    LB_by_TM,
    pred_time,
    try_divide,
)
from areal.experimental.dta.token_trie import TokenTrie
from areal.experimental.dta.trie import CompressedTrie


class ConstantTimeModel:
    def pred(self, stats: dict) -> float:
        return 1.0


class TreeTokenTimeModel:
    def pred(self, stats: dict) -> float:
        return float(stats["n_tree_tokens"])


def _make_seqs() -> list[torch.Tensor]:
    return [
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
        torch.tensor([1, 2, 9], dtype=torch.long),
        torch.tensor([7, 8], dtype=torch.long),
        torch.tensor([7, 8, 9, 10, 11], dtype=torch.long),
    ]


def _assert_partition_valid(bins: list[list[int]], n_items: int, k: int) -> None:
    assert len(bins) == k
    flat = [idx for bucket in bins for idx in bucket]
    assert sorted(flat) == list(range(n_items))


def test_lb_by_n_tokens_assigns_all_sequences_once():
    """LB_by_n_tokens should output a valid partition of original indices."""
    token_seqs = _make_seqs()
    bins = LB_by_n_tokens(token_seqs, K=2)

    _assert_partition_valid(bins, n_items=len(token_seqs), k=2)


def test_lb_by_tm_assigns_all_sequences_once():
    """LB_by_TM should map leaf buckets back to original sequence ids."""
    token_seqs = _make_seqs()
    config = SimpleNamespace(K=2, mode="backward", block_size=2)
    bins = LB_by_TM(token_seqs, ConstantTimeModel(), config)

    _assert_partition_valid(bins, n_items=len(token_seqs), k=2)


def test_try_divide_more_strict_limit_requires_more_partitions():
    """Lower cost_limit should produce at least as many divisions."""
    token_trie = TokenTrie(_make_seqs())
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)
    config = SimpleNamespace(K=2, mode="backward", block_size=2)
    n_seqs = len(token_trie.inputs)
    divL = [0] * (config.K + 1)
    divR = [n_seqs] * (config.K + 1)
    model = TreeTokenTimeModel()

    strict_divs = try_divide(
        compressed_trie, n_seqs, config, divL, divR, model, cost_limit=1.0
    )
    loose_divs = try_divide(
        compressed_trie, n_seqs, config, divL, divR, model, cost_limit=1000.0
    )

    assert len(strict_divs) >= len(loose_divs)


def test_lb_by_dfs_and_tm_assigns_all_sequences_once():
    """LB_by_DFS_and_TM should output a valid partition of original sequence ids."""
    token_seqs = _make_seqs()
    config = SimpleNamespace(K=2, mode="backward", block_size=2)
    bins = LB_by_DFS_and_TM(token_seqs, TreeTokenTimeModel(), config)

    _assert_partition_valid(bins, n_items=len(token_seqs), k=2)


def test_lb_by_dfs_and_tm_k1_returns_single_non_empty_bin():
    """K=1 should place all sequences in the only bucket."""
    token_seqs = _make_seqs()
    config = SimpleNamespace(K=1, mode="backward", block_size=2)
    bins = LB_by_DFS_and_TM(token_seqs, TreeTokenTimeModel(), config)

    assert len(bins) == 1
    assert sorted(bins[0]) == list(range(len(token_seqs)))


def test_lb_by_dfs_and_tm_empty_returns_k_empty_bins():
    """Empty input should return K empty bins without entering search."""
    config = SimpleNamespace(K=3, mode="backward", block_size=2)
    bins = LB_by_DFS_and_TM([], TreeTokenTimeModel(), config)

    assert bins == [[], [], []]


def test_pred_time_rejects_unsupported_mode():
    """pred_time should fail fast on unknown scheduling mode."""
    token_trie = TokenTrie(_make_seqs())
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)

    try:
        pred_time(compressed_trie, ConstantTimeModel(), mode="invalid", block_size=None)
    except ValueError as exc:
        assert "Unsupported mode" in str(exc)
    else:
        raise AssertionError("pred_time should raise ValueError for invalid mode")


def test_lb_by_n_tokens_empty_returns_k_empty_bins():
    """LB_by_n_tokens on an empty input should yield K empty bins."""
    bins = LB_by_n_tokens([], K=3)
    assert bins == [[], [], []]


def test_lb_by_tm_empty_returns_k_empty_bins():
    """LB_by_TM should survive empty inputs now that CompressedTrie does."""
    config = SimpleNamespace(K=3, mode="backward", block_size=2)
    bins = LB_by_TM([], ConstantTimeModel(), config)
    assert bins == [[], [], []]


def test_pred_time_on_empty_trie_returns_finite_value():
    """pred_time should work on an empty compressed trie."""
    trie = TokenTrie([])
    compressed_trie = CompressedTrie(trie.lens, trie.lcp_lens)

    forward_time = pred_time(
        compressed_trie, ConstantTimeModel(), mode="forward", block_size=None
    )
    backward_time = pred_time(
        compressed_trie, ConstantTimeModel(), mode="backward", block_size=2
    )

    assert isinstance(forward_time, float)
    assert isinstance(backward_time, float)
