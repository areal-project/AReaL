# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.weight_update.awex.protocol import (
    ColocateKeyspace,
    ColocateTopology,
)


@pytest.mark.parametrize(
    ("transfer_rank", "engine_rank", "instance_local_rank"),
    [(0, 0, 0), (3, 0, 3), (4, 1, 0), (7, 1, 3)],
)
def test_colocate_topology_decomposes_exact_transfer_rank(
    transfer_rank, engine_rank, instance_local_rank
):
    topology = ColocateTopology(
        transfer_rank=transfer_rank,
        infer_world_size=8,
        train_world_size=8,
        instance_world_size=4,
    )

    assert topology.num_infer_engines == 2
    assert topology.engine_rank == engine_rank
    assert topology.instance_local_rank == instance_local_rank


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            transfer_rank=0,
            infer_world_size=8,
            train_world_size=4,
            instance_world_size=4,
        ),
        dict(
            transfer_rank=0,
            infer_world_size=8,
            train_world_size=8,
            instance_world_size=3,
        ),
        dict(
            transfer_rank=8,
            infer_world_size=8,
            train_world_size=8,
            instance_world_size=4,
        ),
    ],
)
def test_colocate_topology_rejects_invalid_layout(kwargs):
    with pytest.raises(ValueError):
        ColocateTopology(**kwargs)


def test_colocate_keyspace_matches_wire_contract():
    keyspace = ColocateKeyspace("192.0.2.1", 4)

    assert keyspace.writer_version == "awex_writer_version_192.0.2.1_4"
    assert keyspace.serialized_weights(7) == "training_serialized_weights_192.0.2.1_4_7"
    assert keyspace.update_finished(7) == "weights_update_finished_192.0.2.1_4_7"
    assert keyspace.write_finished(7) == "write_finished_192.0.2.1_4_7"
