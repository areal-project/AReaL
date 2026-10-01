"""Regression tests for the empty-input path in TokenTrie and CompressedTrie.

Empty inputs (``inputs == []``) are a legal degenerate state: the trie
contains only the root node and produces empty traversal orders. These
tests pin down that contract so the three permute methods stop raising
``len(lcp_lens) must be ...`` from ``CompressedTrie.__init__``.
"""

import torch

from areal.experimental.dta.token_trie import TokenTrie, _leafization
from areal.experimental.dta.trie import CompressedTrie


def test_token_trie_empty_inputs_construction_is_legal():
    """TokenTrie([]) should build an empty but valid trie."""
    trie = TokenTrie([])

    assert trie.inputs == []
    assert trie.attach_lists == []
    assert trie.lens == []
    assert trie.lcp_lens == []
    assert trie.n_sequences == 0
    assert trie.n_tokens == 0


def test_token_trie_empty_inputs_get_stats_returns_zeros():
    """get_stats on an empty trie returns a well-formed, all-zero summary."""
    trie = TokenTrie([])

    for mode in ("forward", "backward"):
        stats = trie.get_stats(mode=mode, block_size=2)
        assert stats["n_sequences"] == 0
        assert stats["n_tokens"] == 0
        assert stats["n_leaf_sequences"] == 0
        assert stats["n_tree_tokens"] == 0
        assert stats["sum_prefix_len"] == 0
        assert stats["sum_depth"] == 0
        if mode == "backward":
            assert stats["n_f1_tokens"] == 0


def test_token_trie_empty_inputs_permute_is_noop():
    """forward/backward/random permute are no-ops on empty tries, not raises."""
    for permute_name in ("forward_permute", "backward_permute", "random_permute"):
        trie = TokenTrie([])
        getattr(trie, permute_name)()
        assert trie.inputs == []
        assert trie.lens == []
        assert trie.lcp_lens == []


def test_token_trie_explicit_permute_accepts_empty_order():
    """permute([]) must keep an empty trie in the same empty state."""
    trie = TokenTrie([])
    trie.permute([])
    assert trie.inputs == []
    assert trie.attach_lists == []
    assert trie.lens == []
    assert trie.lcp_lens == []


def test_leafization_accepts_empty_lists():
    """_leafization([], []) should return three empty lists without raising."""
    inputs, attach_lists, lcp_lens = _leafization([], [])
    assert inputs == []
    assert attach_lists == []
    assert lcp_lens == []


def test_compressed_trie_empty_inputs_construction_has_only_root():
    """CompressedTrie([], []) is legal and contains just the root node."""
    ct = CompressedTrie([], [])
    assert len(ct.nodes) == 1

    root = ct.nodes[0]
    assert root.depth == 0
    assert root.seq_id == -1
    assert root.child_ids == []


def test_compressed_trie_empty_inputs_rejects_mismatched_lcp_lens():
    """Keep the invariant |lcp_lens| == max(|lens| - 1, 0) enforced."""
    try:
        CompressedTrie([], [0])
    except ValueError as exc:
        assert "len(lcp_lens)" in str(exc)
    else:
        raise AssertionError(
            "CompressedTrie([], [0]) should raise on invariant mismatch"
        )


def test_compressed_trie_empty_inputs_dfs_orders_are_empty():
    """DFS order outputs on an empty trie must all be empty tuples."""
    ct = CompressedTrie([], [])

    order_fwd, lens_fwd, lcp_fwd = ct.get_order_forward()
    assert order_fwd == []
    assert lens_fwd == []
    assert lcp_fwd == []

    order_bwd, lens_bwd, lcp_bwd = ct.get_order_backward()
    assert order_bwd == []
    assert lens_bwd == []
    assert lcp_bwd == []

    order_rnd = ct.get_order_random()
    assert order_rnd == []


def test_token_trie_single_sequence_permute_roundtrip():
    """Sanity check: n == 1 path still works and does not touch ``child_ids[0]``."""
    seq = torch.tensor([1, 2, 3], dtype=torch.long)
    trie = TokenTrie([seq])

    trie.forward_permute()
    assert len(trie.inputs) == 1
    assert torch.equal(trie.inputs[0], seq)
    assert trie.lens == [3]
    assert trie.lcp_lens == []


def test_leafization_absorbs_leading_empty_into_next_leaf():
    """Empty sequences must be absorbed into the next non-empty leaf with length=0.

    Contract for downstream ``DTAEngine``:
        - Leaves themselves are never empty (unless *all* inputs are empty).
        - Empty source sequences survive as ``(attach, 0)`` entries inside the
          merged leaf's ``attach_list`` so we can later decide whether they
          contribute to loss / logprob returns.
    """
    empty = torch.tensor([], dtype=torch.long)
    non_empty = torch.tensor([1, 2, 3], dtype=torch.long)

    empty_att = {"_sequence_batch_id": 0}
    non_empty_att = {"_sequence_batch_id": 1}

    input_ids_leafed, attach_lists, lcp_lens = _leafization(
        [empty, non_empty], [empty_att, non_empty_att]
    )

    assert len(input_ids_leafed) == 1
    assert torch.equal(input_ids_leafed[0], non_empty)
    assert lcp_lens == []

    assert len(attach_lists) == 1
    leaf_attaches = attach_lists[0]
    assert len(leaf_attaches) == 2

    # The leading empty sequence shows up as a length-0 attachment on the leaf.
    (att0, len0), (att1, len1) = leaf_attaches
    assert att0 is empty_att
    assert len0 == 0
    assert att1 is non_empty_att
    assert len1 == 3


def test_token_trie_mixed_empty_keeps_length_zero_entry():
    """TokenTrie public API preserves the (_, 0) attach for empty inputs."""
    empty = torch.tensor([], dtype=torch.long)
    non_empty = torch.tensor([5, 6], dtype=torch.long)

    trie = TokenTrie([empty, non_empty])

    assert len(trie.inputs) == 1
    assert trie.lens == [2]
    assert trie.n_sequences == 2
    assert trie.n_tokens == 2

    lengths = [length for _, length in trie.attach_lists[0]]
    assert 0 in lengths, (
        "Empty source sequence must be preserved as a length-0 entry in "
        "attach_list so DTAEngine can opt it out of loss/logprob computation."
    )
