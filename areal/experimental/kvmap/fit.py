# SPDX-License-Identifier: Apache-2.0
"""Fit a mapper artifact from streamed (source cache, target cache) pairs without holding them all.

Keys are fitted in content space: both source and target keys have their rotary transform
removed before packing, so the fitted map is position free (arXiv:2608.03893, section 3.3).
Values are fitted as stored. Only rows where ``valid`` is true and ``token_index % stride == 0``
enter the accumulators, so padding never trains the map.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from areal.experimental.kvmap.apply import COMPUTE_DTYPE, DenseCache
from areal.experimental.kvmap.artifact import MapperArtifact, ModelIdentity
from areal.experimental.kvmap.contract import FeatureContract, pack_features
from areal.experimental.kvmap.ridge import RidgeAccumulator
from areal.experimental.kvmap.rope import remove_rotary, rotary_cos_sin


@dataclass(frozen=True, kw_only=True)
class CachePair:
    """Native caches of both models for the same token sequences, with positions and a validity mask."""

    source: DenseCache
    target: DenseCache
    positions: torch.Tensor  # [batch, tokens], int64
    valid: torch.Tensor  # [batch, tokens], bool

    def __post_init__(self) -> None:
        shape = (self.source.batch, self.source.tokens)
        if (self.target.batch, self.target.tokens) != shape:
            raise ValueError(
                f"target cache is {(self.target.batch, self.target.tokens)} but source is {shape}"
            )
        if tuple(self.positions.shape) != shape or tuple(self.valid.shape) != shape:
            raise ValueError(
                f"positions {tuple(self.positions.shape)} and valid {tuple(self.valid.shape)} must be {shape}"
            )
        if self.valid.dtype != torch.bool:
            raise ValueError(f"valid must be a bool tensor, got {self.valid.dtype}")


@dataclass(frozen=True, kw_only=True)
class FitOptions:
    contract: FeatureContract
    source: ModelIdentity
    target: ModelIdentity
    lambda_: float
    stride: int
    provenance: dict

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")


def fit_mapper(pairs: Iterable[CachePair], options: FitOptions) -> MapperArtifact:
    """Accumulate every pair, solve one key map and one value map per target layer, and package them."""
    contract = options.contract
    key_accumulators = [
        RidgeAccumulator(
            feature_dim=contract.feature_dim(layer), target_dim=contract.target_dim
        )
        for layer in range(contract.target_num_layers)
    ]
    value_accumulators = [
        RidgeAccumulator(
            feature_dim=contract.feature_dim(layer), target_dim=contract.target_dim
        )
        for layer in range(contract.target_num_layers)
    ]
    pairs_seen = 0
    for pair in pairs:
        _accumulate_pair(pair, options, key_accumulators, value_accumulators)
        pairs_seen += 1
    if pairs_seen == 0:
        raise ValueError("fit_mapper received no cache pairs")
    key_fits = [acc.solve(lambda_=options.lambda_) for acc in key_accumulators]
    value_fits = [acc.solve(lambda_=options.lambda_) for acc in value_accumulators]
    diagnostics = {
        "kind": "ridge",
        "pairs": pairs_seen,
        "rows": key_fits[0].rows,
        "stride": options.stride,
        "key_r_squared_mean_by_layer": [
            float(fit.r_squared.mean()) for fit in key_fits
        ],
        "value_r_squared_mean_by_layer": [
            float(fit.r_squared.mean()) for fit in value_fits
        ],
    }
    return MapperArtifact(
        source=options.source,
        target=options.target,
        contract=contract,
        lambda_=options.lambda_,
        key_weights=tuple(fit.weight.to(torch.float32) for fit in key_fits),
        key_biases=tuple(fit.bias.to(torch.float32) for fit in key_fits),
        value_weights=tuple(fit.weight.to(torch.float32) for fit in value_fits),
        value_biases=tuple(fit.bias.to(torch.float32) for fit in value_fits),
        diagnostics=diagnostics,
        provenance=options.provenance,
    )


def _accumulate_pair(
    pair: CachePair,
    options: FitOptions,
    key_accumulators: list[RidgeAccumulator],
    value_accumulators: list[RidgeAccumulator],
) -> None:
    contract = options.contract
    if (
        len(pair.source.keys) != contract.source_num_layers
        or len(pair.target.keys) != contract.target_num_layers
    ):
        raise ValueError(
            f"pair has {len(pair.source.keys)} source and {len(pair.target.keys)} target layers; contract expects {contract.source_num_layers} and {contract.target_num_layers}"
        )
    tokens = pair.source.tokens
    token_index = (
        torch.arange(tokens, device=pair.valid.device)
        .unsqueeze(0)
        .expand(pair.source.batch, tokens)
    )
    rows = (pair.valid & (token_index % options.stride == 0)).reshape(-1)
    if int(rows.sum()) == 0:
        raise ValueError(
            "a cache pair contributed no valid rows; check the mask and stride"
        )
    source_cos, source_sin = rotary_cos_sin(
        pair.positions,
        head_dim=options.source.head_dim,
        theta=options.source.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    target_cos, target_sin = rotary_cos_sin(
        pair.positions,
        head_dim=options.target.head_dim,
        theta=options.target.rope_theta,
        dtype=COMPUTE_DTYPE,
    )
    source_content = [
        remove_rotary(k.to(COMPUTE_DTYPE), source_cos, source_sin)
        for k in pair.source.keys
    ]
    target_content = [
        remove_rotary(k.to(COMPUTE_DTYPE), target_cos, target_sin)
        for k in pair.target.keys
    ]
    source_values = [v.to(COMPUTE_DTYPE) for v in pair.source.values]
    for layer in range(contract.target_num_layers):
        selected = contract.source_layers_by_target[layer]
        key_accumulators[layer].update(
            pack_features(source_content, selected)[rows],
            pack_features(target_content, (layer,))[rows],
        )
        value_accumulators[layer].update(
            pack_features(source_values, selected)[rows],
            pack_features([v.to(COMPUTE_DTYPE) for v in pair.target.values], (layer,))[
                rows
            ],
        )
