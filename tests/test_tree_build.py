import random

import pytest
import torch

from areal.models.tree_attn.tree import _greedy_build_tries


def _batch(sequences: list[list[int]]) -> dict[str, torch.Tensor]:
    width = max([len(sequence) for sequence in sequences] + [1])
    input_ids = torch.zeros(len(sequences), width, dtype=torch.long)
    attention_mask = torch.zeros(len(sequences), width, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _layout(trie):
    """Describe a trie by its node ids, in pre-order."""
    return (
        [(token, child.start_idx) for token, child in trie.children.items()],
        [
            (
                node.start_idx,
                node.end_idx,
                node.tokens,
                node.sequence_ids,
                [ancestor.start_idx for ancestor in node.ancestors],
                [(token, child.start_idx) for token, child in node.children.items()],
            )
            for node in trie.nodes
        ],
    )


def _new_tokens(tree: dict, sequence: list[int]) -> int:
    node = tree
    for index, token in enumerate(sequence):
        if token not in node["children"]:
            return len(sequence) - index
        node = node["children"][token]
    return 0


def _token_level_layout(tree: dict):
    nodes = []

    def visit(node: dict, ancestors: list[int]) -> None:
        tokens, last = [node["token"]], node
        while len(last["children"]) == 1 and not last["end"]:
            last = next(iter(last["children"].values()))
            tokens.append(last["token"])
        children = sorted(last["children"].items())
        nodes.append(
            (
                node["id"],
                last["id"],
                tokens,
                last["seqs"],
                ancestors,
                [(token, child["id"]) for token, child in children],
            )
        )
        for _, child in children:
            visit(child, ancestors + [node["id"]])

    roots = sorted(tree["children"].items())
    for _, child in roots:
        visit(child, [])
    return [(token, child["id"]) for token, child in roots], nodes


def _token_level_tries(sequences: list[list[int]], capacity: int):
    """Greedy packing with one trie node per token, used as the reference."""
    trees = []
    for seq_id, sequence in enumerate(sequences):
        tree = next(
            (
                tree
                for tree in trees
                if len(tree["nodes"]) + _new_tokens(tree, sequence) <= capacity
            ),
            None,
        )
        if tree is None:
            if len(sequence) > capacity:
                raise ValueError("exceeds max_tokens_per_tree")
            tree = {"children": {}, "nodes": []}
            trees.append(tree)
        node = tree
        for token in sequence:
            if token not in node["children"]:
                child = {
                    "id": len(tree["nodes"]),
                    "token": token,
                    "children": {},
                    "end": False,
                    "seqs": [],
                }
                node["children"][token] = child
                tree["nodes"].append(child)
            node = node["children"][token]
            node["seqs"].append(seq_id)
        node["end"] = True
    return [_token_level_layout(tree) for tree in trees], [
        len(tree["nodes"]) for tree in trees
    ]


def _random_sequences(rng: random.Random) -> list[list[int]]:
    vocab = rng.choice([2, 3, 50])
    bases = [
        [rng.randrange(vocab) for _ in range(rng.randint(0, 24))] for _ in range(3)
    ]
    sequences = []
    for _ in range(rng.randint(1, 24)):
        base = rng.choice(bases)
        tail_len = rng.choice([0, 0, rng.randint(1, 16)])
        sequences.append(
            base[: rng.randint(0, len(base))]
            + [rng.randrange(vocab) for _ in range(tail_len)]
        )
    return sequences


def test_greedy_build_tries_splits_runs_where_sequences_diverge_or_end():
    # Arrange
    sequences = [[1, 2, 3, 4], [1, 2, 5], [1, 2, 3]]

    # Act
    tries, counts = _greedy_build_tries(_batch(sequences), max_tokens_per_tree=8)

    # Assert
    assert counts == [5]
    assert _layout(tries[0]) == (
        [(1, 0)],
        [
            (0, 1, [1, 2], [0, 1, 2], [], [(3, 2), (5, 4)]),
            (2, 2, [3], [0, 2], [0], [(4, 3)]),
            (3, 3, [4], [0], [0, 2], []),
            (4, 4, [5], [1], [0], []),
        ],
    )
    assert all(type(token) is int for node in tries[0].nodes for token in node.tokens)


def test_greedy_build_tries_capacity_counts_only_new_tokens():
    # Arrange
    sequences = [[1, 2, 3], [1, 2, 4], [7, 8]]

    # Act
    tries, counts = _greedy_build_tries(_batch(sequences), max_tokens_per_tree=4)

    # Assert
    assert counts == [4, 2]
    assert [trie.all_sequence_ids for trie in tries] == [[0, 1], [2]]


def test_greedy_build_tries_sequence_over_capacity_raises():
    with pytest.raises(ValueError, match="exceeds max_tokens_per_tree"):
        _greedy_build_tries(_batch([[1, 2, 3]]), max_tokens_per_tree=2)


@pytest.mark.parametrize("seed", range(4))
def test_greedy_build_tries_random_batches_match_token_level_reference(seed):
    rng = random.Random(seed)
    for _ in range(200):
        # Arrange
        sequences = _random_sequences(rng)
        longest = max(len(sequence) for sequence in sequences)
        capacity = rng.choice(
            [
                max(longest, 1),
                longest + rng.randint(0, 16),
                sum(len(sequence) for sequence in sequences) + 1,
            ]
        )

        # Act
        expected_layouts, expected_counts = _token_level_tries(sequences, capacity)
        tries, counts = _greedy_build_tries(_batch(sequences), capacity)

        # Assert
        assert counts == expected_counts
        assert [_layout(trie) for trie in tries] == expected_layouts
