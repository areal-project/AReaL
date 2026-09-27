# SPDX-License-Identifier: Apache-2.0
"""A fitted mapper recovers a planted content-space map, learns identity on identical caches, and ignores masked rows."""

import pytest
import torch

from tests.experimental.kvmap.conftest import (
    BATCH,
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    SOURCE_SHA,
    TARGET_SHA,
    TOKENS,
    identity,
    positions,
    random_cache,
)

from areal.experimental.kvmap.apply import DenseCache, translate_cache
from areal.experimental.kvmap.contract import (
    FeatureContract,
    pack_features,
    unpack_target,
)
from areal.experimental.kvmap.fit import CachePair, FitOptions, fit_mapper
from areal.experimental.kvmap.rope import apply_rotary, remove_rotary, rotary_cos_sin

SOURCE_THETA = 10_000.0
TARGET_THETA = 1_000_000.0


def _options(**overrides) -> FitOptions:
    base = dict(
        contract=FeatureContract.same_layer(
            num_layers=LAYERS, kv_heads=KV_HEADS, head_dim=HEAD_DIM
        ),
        source=identity(SOURCE_SHA, theta=SOURCE_THETA),
        target=identity(TARGET_SHA, theta=TARGET_THETA),
        lambda_=1e-6,
        stride=1,
        provenance={"test": True},
    )
    base.update(overrides)
    return FitOptions(**base)


def _planted_target(
    source: DenseCache, weights, biases, value_weights, value_biases
) -> DenseCache:
    pos = positions(source.batch, source.tokens)
    cos_s, sin_s = rotary_cos_sin(
        pos, head_dim=HEAD_DIM, theta=SOURCE_THETA, dtype=torch.float32
    )
    cos_t, sin_t = rotary_cos_sin(
        pos, head_dim=HEAD_DIM, theta=TARGET_THETA, dtype=torch.float32
    )
    keys, values = [], []
    for layer in range(LAYERS):
        content = (
            pack_features([remove_rotary(source.keys[layer], cos_s, sin_s)], (0,))
            @ weights[layer]
            + biases[layer]
        )
        keys.append(
            apply_rotary(
                unpack_target(
                    content,
                    batch=source.batch,
                    tokens=source.tokens,
                    kv_heads=KV_HEADS,
                    head_dim=HEAD_DIM,
                ),
                cos_t,
                sin_t,
            )
        )
        rows = (
            pack_features([source.values[layer]], (0,)) @ value_weights[layer]
            + value_biases[layer]
        )
        values.append(
            unpack_target(
                rows,
                batch=source.batch,
                tokens=source.tokens,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
            )
        )
    return DenseCache(keys=tuple(keys), values=tuple(values))


def _planted_maps(seed: int):
    generator = torch.Generator().manual_seed(seed)
    dim = KV_HEADS * HEAD_DIM

    def make():
        return [
            torch.randn(dim, dim, generator=generator) / dim**0.5 for _ in range(LAYERS)
        ]

    def make_bias():
        return [torch.randn(dim, generator=generator) for _ in range(LAYERS)]

    return make(), make_bias(), make(), make_bias()


def test_planted_content_space_map_is_recovered_on_held_out_caches():
    maps = _planted_maps(0)
    pairs = [
        CachePair(
            source=random_cache(seed),
            target=_planted_target(random_cache(seed), *maps),
            positions=positions(),
            valid=torch.ones(BATCH, TOKENS, dtype=torch.bool),
        )
        for seed in range(1, 5)
    ]
    artifact = fit_mapper(pairs, _options())

    held_out = random_cache(99)
    predicted = translate_cache(
        source=held_out,
        positions=positions(),
        artifact=artifact,
        output_dtype=torch.float32,
    )
    expected = _planted_target(held_out, *maps)

    for layer in range(LAYERS):
        torch.testing.assert_close(
            predicted.keys[layer], expected.keys[layer], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            predicted.values[layer], expected.values[layer], rtol=1e-4, atol=1e-4
        )
    assert min(artifact.diagnostics["key_r_squared_mean_by_layer"]) > 0.999


def test_identical_caches_learn_a_near_identity_map():
    pairs = [
        CachePair(
            source=random_cache(seed),
            target=random_cache(seed),
            positions=positions(),
            valid=torch.ones(BATCH, TOKENS, dtype=torch.bool),
        )
        for seed in range(1, 5)
    ]
    artifact = fit_mapper(
        pairs,
        _options(
            source=identity(SOURCE_SHA), target=identity(TARGET_SHA), lambda_=0.01
        ),
    )

    held_out = random_cache(7)
    out = translate_cache(
        source=held_out,
        positions=positions(),
        artifact=artifact,
        output_dtype=torch.float32,
    )

    for layer in range(LAYERS):
        torch.testing.assert_close(
            out.keys[layer], held_out.keys[layer], rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            out.values[layer], held_out.values[layer], rtol=1e-3, atol=1e-3
        )


def test_masked_rows_do_not_influence_the_fit():
    maps = _planted_maps(1)
    valid = torch.ones(BATCH, TOKENS, dtype=torch.bool)
    valid[:, TOKENS // 2 :] = False
    clean = [
        CachePair(
            source=random_cache(s),
            target=_planted_target(random_cache(s), *maps),
            positions=positions(),
            valid=valid,
        )
        for s in range(1, 4)
    ]
    poisoned = []
    for pair in clean:
        keys = tuple(k.clone() for k in pair.target.keys)
        for k in keys:
            k[:, :, TOKENS // 2 :, :] = 1e6
        poisoned.append(
            CachePair(
                source=pair.source,
                target=DenseCache(keys=keys, values=pair.target.values),
                positions=pair.positions,
                valid=valid,
            )
        )

    clean_fit = fit_mapper(clean, _options())
    poisoned_fit = fit_mapper(poisoned, _options())

    for layer in range(LAYERS):
        assert torch.equal(
            clean_fit.key_weights[layer], poisoned_fit.key_weights[layer]
        )
    assert clean_fit.diagnostics["rows"] == BATCH * TOKENS // 2 * 3


def test_stride_reduces_rows_and_all_masked_pair_is_rejected():
    pair = CachePair(
        source=random_cache(1),
        target=random_cache(1),
        positions=positions(),
        valid=torch.ones(BATCH, TOKENS, dtype=torch.bool),
    )
    assert (
        fit_mapper([pair], _options(stride=4)).diagnostics["rows"]
        == BATCH * TOKENS // 4
    )
    empty = CachePair(
        source=random_cache(1),
        target=random_cache(1),
        positions=positions(),
        valid=torch.zeros(BATCH, TOKENS, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="no valid rows"):
        fit_mapper([empty], _options())
    with pytest.raises(ValueError, match="no cache pairs"):
        fit_mapper([], _options())
