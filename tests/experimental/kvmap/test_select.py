# SPDX-License-Identifier: Apache-2.0
"""Top-k selection finds the source layer that a target layer was planted to depend on."""

import pytest
import torch

from tests.experimental.kvmap.conftest import (
    BATCH,
    LAYERS,
    SOURCE_SHA,
    TARGET_SHA,
    TOKENS,
    identity,
    positions,
    random_cache,
)

from areal.experimental.kvmap.apply import DenseCache
from areal.experimental.kvmap.fit import CachePair
from areal.experimental.kvmap.select import select_source_layers_by_r_squared


def _shifted_target(source: DenseCache) -> DenseCache:
    # target layer l is a copy of source layer (l + 1) % LAYERS, so the best single source is unambiguous
    order = [(layer + 1) % LAYERS for layer in range(LAYERS)]
    return DenseCache(
        keys=tuple(source.keys[i] for i in order),
        values=tuple(source.values[i] for i in order),
    )


def test_k1_selection_recovers_the_planted_dependency():
    pairs = [
        CachePair(
            source=random_cache(s),
            target=_shifted_target(random_cache(s)),
            positions=positions(),
            valid=torch.ones(BATCH, TOKENS, dtype=torch.bool),
        )
        for s in range(1, 3)
    ]
    contract = select_source_layers_by_r_squared(
        pairs,
        source=identity(SOURCE_SHA),
        target=identity(TARGET_SHA),
        k=1,
        lambda_=0.01,
        stride=1,
    )
    assert contract.source_layers_by_target == tuple(
        ((layer + 1) % LAYERS,) for layer in range(LAYERS)
    )


def test_k_equal_all_layers_selects_everything_sorted():
    pairs = [
        CachePair(
            source=random_cache(1),
            target=random_cache(2),
            positions=positions(),
            valid=torch.ones(BATCH, TOKENS, dtype=torch.bool),
        )
    ]
    contract = select_source_layers_by_r_squared(
        pairs,
        source=identity(SOURCE_SHA),
        target=identity(TARGET_SHA),
        k=LAYERS,
        lambda_=0.01,
        stride=2,
    )
    assert all(
        selection == tuple(range(LAYERS))
        for selection in contract.source_layers_by_target
    )


@pytest.mark.parametrize("k", [0, LAYERS + 1])
def test_out_of_range_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must be in"):
        select_source_layers_by_r_squared(
            [],
            source=identity(SOURCE_SHA),
            target=identity(TARGET_SHA),
            k=k,
            lambda_=0.01,
            stride=1,
        )
