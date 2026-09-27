# SPDX-License-Identifier: Apache-2.0
"""An artifact round-trips through disk unchanged and refuses any pair it was not fitted for."""

import json
import pathlib

import pytest
import torch

from tests.experimental.kvmap.conftest import (
    SOURCE_SHA,
    TARGET_SHA,
    identity,
    positions,
    random_cache,
)

from areal.experimental.kvmap.apply import translate_cache
from areal.experimental.kvmap.artifact import (
    META_FILE,
    MapperArtifact,
    MapperMismatchError,
    identity_artifact,
)


def test_identity_artifact_is_a_bypass_within_float32_rotary_round_trip():
    artifact = identity_artifact(
        source=identity(SOURCE_SHA),
        target=identity(TARGET_SHA),
        provenance={"test": True},
    )
    cache = random_cache(0)

    out = translate_cache(
        source=cache,
        positions=positions(),
        artifact=artifact,
        output_dtype=torch.float32,
    )

    for layer in range(len(cache.keys)):
        torch.testing.assert_close(
            out.keys[layer], cache.keys[layer], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            out.values[layer], cache.values[layer], rtol=0.0, atol=0.0
        )


def test_identity_artifact_with_different_target_theta_re_rotates_keys():
    artifact = identity_artifact(
        source=identity(SOURCE_SHA, theta=10_000.0),
        target=identity(TARGET_SHA, theta=1_000_000.0),
        provenance={},
    )
    cache = random_cache(1)
    out = translate_cache(
        source=cache,
        positions=positions(),
        artifact=artifact,
        output_dtype=torch.float32,
    )
    assert not torch.allclose(out.keys[0], cache.keys[0], atol=1e-3)
    torch.testing.assert_close(out.values[0], cache.values[0], rtol=0.0, atol=0.0)


def test_save_then_load_round_trips_and_refuses_overwrite(tmp_path: pathlib.Path):
    artifact = identity_artifact(
        source=identity(SOURCE_SHA),
        target=identity(TARGET_SHA),
        provenance={"commit": "deadbeef"},
    )
    destination = tmp_path / "mappers" / "a-b"

    sha = artifact.save(destination)
    loaded = MapperArtifact.load(destination)

    assert len(sha) == 64
    assert (
        loaded.source == artifact.source
        and loaded.target == artifact.target
        and loaded.contract == artifact.contract
    )
    assert loaded.provenance == {"commit": "deadbeef"}
    for layer in range(artifact.contract.target_num_layers):
        assert torch.equal(loaded.key_weights[layer], artifact.key_weights[layer])
        assert torch.equal(loaded.value_biases[layer], artifact.value_biases[layer])
    with pytest.raises(FileExistsError, match="immutable"):
        artifact.save(destination)
    assert sorted(p.name for p in destination.parent.iterdir()) == ["a-b"]


def test_load_rejects_unknown_meta_key_and_wrong_format_version(tmp_path: pathlib.Path):
    artifact = identity_artifact(
        source=identity(SOURCE_SHA), target=identity(TARGET_SHA), provenance={}
    )
    destination = tmp_path / "a-b"
    artifact.save(destination)
    meta_path = destination / META_FILE
    meta = json.loads(meta_path.read_text())
    meta["extra"] = 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="must be exactly"):
        MapperArtifact.load(destination)
    del meta["extra"]
    meta["format_version"] = 99
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version 99"):
        MapperArtifact.load(destination)


def test_validate_pair_names_the_first_differing_field():
    artifact = identity_artifact(
        source=identity(SOURCE_SHA), target=identity(TARGET_SHA), provenance={}
    )
    artifact.validate_pair(source=identity(SOURCE_SHA), target=identity(TARGET_SHA))
    with pytest.raises(MapperMismatchError, match="target.checkpoint_sha256"):
        artifact.validate_pair(source=identity(SOURCE_SHA), target=identity("c" * 64))
    with pytest.raises(MapperMismatchError, match="source.rope_theta"):
        artifact.validate_pair(
            source=identity(SOURCE_SHA, theta=10_000.0), target=identity(TARGET_SHA)
        )


def test_artifact_rejects_weight_shape_that_disagrees_with_contract():
    artifact = identity_artifact(
        source=identity(SOURCE_SHA), target=identity(TARGET_SHA), provenance={}
    )
    bad = artifact.key_weights[:-1] + (torch.eye(3),)
    with pytest.raises(ValueError, match="key weight for target layer 2 has shape"):
        MapperArtifact(
            source=artifact.source,
            target=artifact.target,
            contract=artifact.contract,
            lambda_=0.0,
            key_weights=bad,
            key_biases=artifact.key_biases,
            value_weights=artifact.value_weights,
            value_biases=artifact.value_biases,
            diagnostics={},
            provenance={},
        )
