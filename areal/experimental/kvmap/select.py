# SPDX-License-Identifier: Apache-2.0
"""Choose the top-k source layers per target layer by single-layer ridge R^2 (arXiv:2608.03893, section 3.2).

Score(target l, source s) = mean of the in-sample R^2 of a same-width ridge fit from source
layer s to target layer l, averaged over keys (content space) and values. Ties break toward
the lower source index; the selection is returned sorted ascending.
"""

from collections.abc import Sequence

import torch

from areal.experimental.kvmap.apply import COMPUTE_DTYPE
from areal.experimental.kvmap.artifact import ModelIdentity
from areal.experimental.kvmap.contract import FeatureContract, pack_features
from areal.experimental.kvmap.fit import CachePair
from areal.experimental.kvmap.ridge import RidgeAccumulator
from areal.experimental.kvmap.rope import remove_rotary, rotary_cos_sin


def select_source_layers_by_r_squared(
    pairs: Sequence[CachePair],
    *,
    source: ModelIdentity,
    target: ModelIdentity,
    k: int,
    lambda_: float,
    stride: int,
) -> FeatureContract:
    """Fit every (source layer, target layer) pair once and keep the ``k`` best sources per target."""
    if k < 1 or k > source.num_layers:
        raise ValueError(f"k must be in [1, {source.num_layers}], got {k}")
    if len(pairs) == 0:
        raise ValueError("selection needs at least one cache pair")
    source_dim = source.num_kv_heads * source.head_dim
    target_dim = target.num_kv_heads * target.head_dim
    keys = [
        [
            RidgeAccumulator(feature_dim=source_dim, target_dim=target_dim)
            for _ in range(source.num_layers)
        ]
        for _ in range(target.num_layers)
    ]
    values = [
        [
            RidgeAccumulator(feature_dim=source_dim, target_dim=target_dim)
            for _ in range(source.num_layers)
        ]
        for _ in range(target.num_layers)
    ]
    for pair in pairs:
        _accumulate(
            pair, source=source, target=target, stride=stride, keys=keys, values=values
        )
    scores = torch.zeros(target.num_layers, source.num_layers, dtype=torch.float64)
    for target_layer in range(target.num_layers):
        for source_layer in range(source.num_layers):
            scores[target_layer, source_layer] = 0.5 * (
                float(
                    keys[target_layer][source_layer]
                    .solve(lambda_=lambda_)
                    .r_squared.mean()
                )
                + float(
                    values[target_layer][source_layer]
                    .solve(lambda_=lambda_)
                    .r_squared.mean()
                )
            )
    selection = []
    for target_layer in range(target.num_layers):
        order = sorted(
            range(source.num_layers), key=lambda s: (-float(scores[target_layer, s]), s)
        )
        selection.append(tuple(sorted(order[:k])))
    return FeatureContract(
        source_layers_by_target=tuple(selection),
        source_num_layers=source.num_layers,
        source_kv_heads=source.num_kv_heads,
        source_head_dim=source.head_dim,
        target_kv_heads=target.num_kv_heads,
        target_head_dim=target.head_dim,
    )


def _accumulate(
    pair: CachePair,
    *,
    source: ModelIdentity,
    target: ModelIdentity,
    stride: int,
    keys,
    values,
) -> None:
    tokens = pair.source.tokens
    token_index = (
        torch.arange(tokens, device=pair.valid.device)
        .unsqueeze(0)
        .expand(pair.source.batch, tokens)
    )
    rows = (pair.valid & (token_index % stride == 0)).reshape(-1)
    source_cos, source_sin = rotary_cos_sin(
        pair.positions,
        head_dim=source.head_dim,
        theta=source.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    target_cos, target_sin = rotary_cos_sin(
        pair.positions,
        head_dim=target.head_dim,
        theta=target.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    source_content = [
        pack_features(
            [remove_rotary(k.to(COMPUTE_DTYPE), source_cos, source_sin)], (0,)
        )[rows]
        for k in pair.source.keys
    ]
    target_content = [
        pack_features(
            [remove_rotary(k.to(COMPUTE_DTYPE), target_cos, target_sin)], (0,)
        )[rows]
        for k in pair.target.keys
    ]
    source_values = [
        pack_features([v.to(COMPUTE_DTYPE)], (0,))[rows] for v in pair.source.values
    ]
    target_values = [
        pack_features([v.to(COMPUTE_DTYPE)], (0,))[rows] for v in pair.target.values
    ]
    for target_layer in range(target.num_layers):
        for source_layer in range(source.num_layers):
            keys[target_layer][source_layer].update(
                source_content[source_layer], target_content[target_layer]
            )
            values[target_layer][source_layer].update(
                source_values[source_layer], target_values[target_layer]
            )
