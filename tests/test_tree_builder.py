import random

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn.tree import (
    TrieNode,
    _greedy_build_tries,
    build_packed_tree_batch,
    trie_to_parent_array,
)


class _RefBuildNode:
    """Reference per-token node mirroring the pre-refactor implementation."""

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _RefBuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _ref_count_additional_nodes(root: _RefBuildNode, sequence: list[int]) -> int:
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _ref_insert_sequence(
    root: _RefBuildNode,
    all_nodes: list[_RefBuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _RefBuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


def _ref_compress_trie(root: _RefBuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(
        node: _RefBuildNode,
        ancestors: list[TrieNode],
    ) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id

        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            current = next(iter(current.children.values()))

        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)

        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(
                    child,
                    ancestors + [trie_node],
                )

        return trie_node

    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])

    return trie_root


def _ref_greedy_build_tries(
    data: dict[str, torch.Tensor],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    sequences = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(data["input_ids"], data["attention_mask"])
    ]
    forests: list[dict] = []

    for seq_id, seq in enumerate(sequences):
        inserted = False
        for tree_id, tree in enumerate(forests):
            additional = _ref_count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _ref_insert_sequence(
                    tree["root"],
                    tree["all_nodes"],
                    seq,
                    tree_id,
                    seq_id,
                )
                tree["nodes"] += additional
                inserted = True
                break

        if inserted:
            continue

        if len(seq) > max_tokens_per_tree:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_tokens_per_tree "
                f"{max_tokens_per_tree}; adjust limit or split sequences."
            )

        new_tree_id = len(forests)
        new_root = _RefBuildNode(new_tree_id, -1, -1)
        all_nodes: list[_RefBuildNode] = []
        _ref_insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    tries = [_ref_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]
    return tries, num_tokens_list


def _make_padded_batch(
    sequences: list[list[int]], dtype: torch.dtype = torch.long
) -> dict[str, torch.Tensor]:
    if not sequences:
        return {
            "input_ids": torch.empty((0, 0), dtype=dtype),
            "attention_mask": torch.empty((0, 0), dtype=torch.bool),
        }
    max_len = max((len(s) for s in sequences), default=0)
    input_ids = torch.zeros((len(sequences), max_len), dtype=dtype)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for idx, seq in enumerate(sequences):
        if seq:
            input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=dtype)
            attention_mask[idx, : len(seq)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _assert_subtree_equal(actual: TrieNode, ref: TrieNode) -> None:
    assert actual.tree_id == ref.tree_id
    assert actual.start_idx == ref.start_idx
    assert actual.end_idx == ref.end_idx
    assert actual.tokens == ref.tokens
    assert all(type(t) is int for t in actual.tokens)
    assert actual.sequence_ids == ref.sequence_ids
    assert [a.tree_indices for a in actual.ancestors] == [
        a.tree_indices for a in ref.ancestors
    ]
    assert list(actual.children.keys()) == list(ref.children.keys())
    assert all(type(k) is int for k in actual.children)
    for k in ref.children:
        _assert_subtree_equal(actual.children[k], ref.children[k])


def _assert_tries_equal(
    sequences: list[list[int]],
    max_tokens_per_tree: int,
    dtype: torch.dtype = torch.long,
) -> None:
    data = _make_padded_batch(sequences, dtype=dtype)
    actual_tries, actual_counts = _greedy_build_tries(data, max_tokens_per_tree)
    ref_tries, ref_counts = _ref_greedy_build_tries(data, max_tokens_per_tree)

    assert actual_counts == ref_counts
    assert len(actual_tries) == len(ref_tries)

    for actual, ref in zip(actual_tries, ref_tries, strict=True):
        _assert_subtree_equal(actual, ref)
        assert actual.all_sequence_ids == ref.all_sequence_ids
        assert len(actual.nodes) == len(ref.nodes)
        for a_node, r_node in zip(actual.nodes, ref.nodes, strict=True):
            assert a_node.tree_id == r_node.tree_id
            assert a_node.tree_indices == r_node.tree_indices
            assert a_node.tokens == r_node.tokens
            assert a_node.sequence_ids == r_node.sequence_ids
            assert [a.tree_indices for a in a_node.ancestors] == [
                a.tree_indices for a in r_node.ancestors
            ]
            assert list(a_node.children.keys()) == list(r_node.children.keys())
        torch.testing.assert_close(
            trie_to_parent_array(actual, max_tokens_per_tree),
            trie_to_parent_array(ref, max_tokens_per_tree),
        )


def test_trie_builder_equivalence_multi_turn_extensions():
    sys_prompt = list(range(100, 140))
    sequences = [
        sys_prompt + [1, 2, 3],
        sys_prompt + [1, 2, 3, 4, 5, 6],
        sys_prompt + [1, 2, 3, 4, 5, 6, 7, 8],
        sys_prompt + [1, 2, 9, 10],
        sys_prompt + [20, 21],
    ]
    _assert_tries_equal(sequences, max_tokens_per_tree=128)


def test_trie_builder_equivalence_splits_and_prefixes():
    # Exercises: long-then-short mid-run split, short-then-long extension,
    # repeated splitting of the same suffix run, duplicate sequences,
    # empty sequences, and single-token sequences.
    sequences = [
        [10, 20, 30, 40, 50, 60],
        [10, 20, 30, 40],  # ends mid-run -> splits [10..40] / [50, 60]
        [10, 20, 99, 100],  # splits [10, 20] / [30, 40] + adds [99, 100]
        [10, 20, 30, 40, 50, 60],  # exact duplicate along split chain
        [10, 20, 30],  # splits [30, 40] again
        [10],  # single-token prefix
        [10, 20, 30, 40, 50, 60, 70],  # extends existing leaf
        [],  # empty sequence
        [1],  # disjoint root branch
        [],  # second empty sequence
    ]
    _assert_tries_equal(sequences, max_tokens_per_tree=32)


def test_trie_builder_equivalence_dtypes_and_sparse_masks():
    sequences = [
        [5, 6, 7, 8, 9],
        [5, 6, 10, 11],
        [5, 6, 7, 12],
    ]
    _assert_tries_equal(sequences, max_tokens_per_tree=16, dtype=torch.int32)

    # Non-contiguous masks and empty-batch edge cases
    data = {
        "input_ids": torch.tensor(
            [
                [999, 1, 2, 999, 3, 4],
                [888, 1, 2, 5, 6, 888],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [False, True, True, False, True, True],
                [False, True, True, True, True, False],
                [False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    }
    actual_tries, actual_counts = _greedy_build_tries(data, max_tokens_per_tree=16)
    ref_tries, ref_counts = _ref_greedy_build_tries(data, max_tokens_per_tree=16)
    assert actual_counts == ref_counts
    for a, r in zip(actual_tries, ref_tries, strict=True):
        _assert_subtree_equal(a, r)

    empty_tries, empty_counts = _greedy_build_tries(
        _make_padded_batch([]), max_tokens_per_tree=16
    )
    assert empty_tries == []
    assert empty_counts == []


def test_trie_builder_random_fuzz_multiple_trees():
    rng = random.Random(2026)
    vocab_prompts = [
        [rng.randint(1, 50) for _ in range(rng.randint(8, 20))] for _ in range(6)
    ]
    for _trial in range(25):
        sequences: list[list[int]] = []
        for _ in range(20):
            base = rng.choice(vocab_prompts)
            cut = rng.randint(1, len(base))
            suffix = [rng.randint(1, 12) for _ in range(rng.randint(0, 18))]
            sequences.append(base[:cut] + suffix)
        _assert_tries_equal(sequences, max_tokens_per_tree=64)


def test_trie_builder_exact_boundary_and_overflow_error():
    exact = [[1, 2, 3, 4], [1, 2, 3, 4]]
    _assert_tries_equal(exact, max_tokens_per_tree=4)

    overflow = _make_padded_batch([[1, 2, 3, 4, 5]])
    with pytest.raises(ValueError, match="exceeds max_tokens_per_tree"):
        _greedy_build_tries(overflow, max_tokens_per_tree=4)


def test_build_packed_tree_batch_end_to_end_cpu():
    data = _make_padded_batch(
        [
            [10, 11, 12, 13],
            [10, 11, 14, 15],
            [20, 21, 22],
        ]
    )
    data["loss_mask"] = data["attention_mask"].clone()
    mb_list = build_packed_tree_batch(
        data,
        MicroBatchSpec(max_tokens_per_mb=128),
        pad_to_maximum=True,
    )
    assert len(mb_list.mbs) == 1
    assert mb_list.group_lens == [9]
    trie = mb_list.mbs[0]["trie_node"]
    assert isinstance(trie, TrieNode)
    assert trie.num_tokens == 9
    assert sorted(trie.all_sequence_ids) == [0, 1, 2]
