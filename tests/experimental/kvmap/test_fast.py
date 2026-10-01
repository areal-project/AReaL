# SPDX-License-Identifier: Apache-2.0
"""The stacked mapper equals the per-layer reference translation, for mixed positions, and refuses other contracts."""

import pytest
import torch

from areal.experimental.kvmap.apply import DenseCache, translate_cache
from areal.experimental.kvmap.artifact import MapperArtifact
from areal.experimental.kvmap.contract import FeatureContract
from areal.experimental.kvmap.fast import StackedMapper
from tests.experimental.kvmap.conftest import HEAD_DIM, KV_HEADS, LAYERS, SOURCE_SHA, TARGET_SHA, identity

WIDTH = KV_HEADS * HEAD_DIM


def _random_artifact(seed: int, contract: FeatureContract | None = None) -> MapperArtifact:
    generator = torch.Generator().manual_seed(seed)
    contract = contract or FeatureContract.same_layer(num_layers=LAYERS, kv_heads=KV_HEADS, head_dim=HEAD_DIM)

    def weights():
        return tuple(torch.randn(contract.feature_dim(layer), WIDTH, generator=generator) / WIDTH**0.5 for layer in range(LAYERS))

    def biases():
        return tuple(torch.randn(WIDTH, generator=generator) for _ in range(LAYERS))

    return MapperArtifact(source=identity(SOURCE_SHA, theta=10_000.0), target=identity(TARGET_SHA), contract=contract, lambda_=0.01,
                          key_weights=weights(), key_biases=biases(), value_weights=weights(), value_biases=biases(), diagnostics={}, provenance={})


def test_stacked_translation_equals_reference_for_two_concatenated_sequences():
    artifact = _random_artifact(0)
    generator = torch.Generator().manual_seed(1)
    lengths = (5, 3)
    positions = torch.cat([torch.arange(n) for n in lengths])  # two sequences, positions restart
    tokens = int(positions.shape[0])
    keys = torch.randn(LAYERS, tokens, KV_HEADS, HEAD_DIM, generator=generator)
    values = torch.randn(LAYERS, tokens, KV_HEADS, HEAD_DIM, generator=generator)

    fast_keys, fast_values = StackedMapper(artifact, device="cpu").translate(keys=keys, values=values, positions=positions)

    # Reference: the per-layer path on [1, kv_heads, tokens, d] with the same explicit positions.
    reference = translate_cache(
        source=DenseCache(keys=tuple(keys[layer].permute(1, 0, 2).unsqueeze(0) for layer in range(LAYERS)),
                          values=tuple(values[layer].permute(1, 0, 2).unsqueeze(0) for layer in range(LAYERS))),
        positions=positions.unsqueeze(0), artifact=artifact, output_dtype=torch.float32)
    for layer in range(LAYERS):
        torch.testing.assert_close(fast_keys[layer], reference.keys[layer][0].permute(1, 0, 2), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(fast_values[layer], reference.values[layer][0].permute(1, 0, 2), rtol=1e-5, atol=1e-5)
    # Witness that the map is not the identity, so agreement is informative.
    assert not torch.allclose(fast_keys, keys, atol=1e-2)


def test_positions_beyond_the_initial_table_are_handled():
    artifact = _random_artifact(2)
    mapper = StackedMapper(artifact, device="cpu")
    positions = torch.tensor([0, 4095, 4096, 20_000])
    keys = torch.randn(LAYERS, 4, KV_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(3))
    out, _ = mapper.translate(keys=keys, values=keys, positions=positions)
    reference = translate_cache(source=DenseCache(keys=tuple(keys[l].permute(1, 0, 2).unsqueeze(0) for l in range(LAYERS)), values=tuple(keys[l].permute(1, 0, 2).unsqueeze(0) for l in range(LAYERS))),
                                positions=positions.unsqueeze(0), artifact=artifact, output_dtype=torch.float32)
    torch.testing.assert_close(out[0], reference.keys[0][0].permute(1, 0, 2), rtol=1e-4, atol=1e-4)


def test_output_keeps_the_pool_dtype_and_empty_input_is_a_no_op():
    mapper = StackedMapper(_random_artifact(4), device="cpu")
    keys = torch.randn(LAYERS, 2, KV_HEADS, HEAD_DIM).to(torch.bfloat16)
    out_keys, out_values = mapper.translate(keys=keys, values=keys, positions=torch.arange(2))
    assert out_keys.dtype == torch.bfloat16 and out_values.dtype == torch.bfloat16
    empty = torch.empty(LAYERS, 0, KV_HEADS, HEAD_DIM)
    assert mapper.translate(keys=empty, values=empty, positions=torch.empty(0, dtype=torch.long))[0].shape[1] == 0


def test_cross_layer_artifacts_and_wrong_shapes_are_rejected():
    cross = FeatureContract(source_layers_by_target=((0, 1), (1, 2), (0, 2)), source_num_layers=LAYERS, source_kv_heads=KV_HEADS,
                            source_head_dim=HEAD_DIM, target_kv_heads=KV_HEADS, target_head_dim=HEAD_DIM)
    with pytest.raises(ValueError, match="same-layer artifacts only"):
        StackedMapper(_random_artifact(5, cross), device="cpu")
    mapper = StackedMapper(_random_artifact(6), device="cpu")
    with pytest.raises(ValueError, match="layers, tokens, kv_heads, head_dim"):
        mapper.translate(keys=torch.zeros(LAYERS, 3, KV_HEADS, HEAD_DIM), values=torch.zeros(LAYERS, 3, KV_HEADS, HEAD_DIM), positions=torch.arange(2))
