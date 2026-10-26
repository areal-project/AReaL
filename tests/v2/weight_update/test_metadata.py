# SPDX-License-Identifier: Apache-2.0
"""Normal DP/TP publication plans preserve every physical destination shard."""

from collections import Counter
from math import prod

import pytest
import torch

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.v2.weight_update.gateway.metadata import (
    inference_metadata_template,
    merge_parameter_metadata,
)

awex = pytest.importorskip("awex.meta.weight_meta")
ShardingType = pytest.importorskip("awex.sharding.param_sharding").ShardingType
TransferPlanBuilder = pytest.importorskip(
    "awex.transfer.transfer_plan"
).TransferPlanBuilder


def parameter(name, rank, offset, shape, global_shape):
    shard = awex.ParameterShardMeta(
        name=name,
        tp_rank=rank,
        attn_tp_rank=rank,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=rank,
        world_size=8,
        engine_rank=0,
        shape=shape,
        numel=prod(shape),
        dtype=torch.bfloat16,
        global_offset=offset,
        sharding_type=ShardingType.NO_SHARDING,
        num_shards=1,
        sharding_dim=0,
    )
    return awex.ParameterMeta(
        name=name,
        global_numel=prod(global_shape),
        global_shape=global_shape,
        dtype=torch.bfloat16,
        shards=[shard],
        replicas=[awex.ParameterReplicaMeta(shards=[shard])],
    )


@pytest.mark.parametrize(
    "actors,tp,engines", [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2), (4, 2, 2)]
)
def test_dense_dp_tp_plan_covers_each_receiver_once(actors, tp, engines):
    training = [
        parameter(name, rank, offset, shape, shape)
        for rank in range(actors)
        for name, offset, shape in (("weight", (0, 0), (4, 4)), ("norm", (0,), (4,)))
    ]
    inference = [
        item
        for rank in range(tp)
        for item in (
            parameter("weight", rank, (rank * (4 // tp), 0), (4 // tp, 4), (4, 4)),
            parameter("norm", rank, (0,), (4,), (4,)),
        )
    ]
    train_meta = deserialize_value(merge_parameter_metadata(serialize_value(training)))
    infer_meta = deserialize_value(
        inference_metadata_template(
            [{"result": serialize_value(inference)} for _ in range(engines)]
        )
    )
    operations = TransferPlanBuilder(
        tp * engines, actors, engines
    ).build_weights_mapping_operations(infer_meta, train_meta)
    counts = Counter((op.recv_rank, op.recv_shard_meta.name) for op in operations)
    assert counts == Counter(
        {(rank, name): 1 for rank in range(tp * engines) for name in ("weight", "norm")}
    )
    received = Counter()
    for operation in operations:
        received[operation.recv_rank] += prod(operation.overlap_shape) * 2
    assert received == Counter(
        {rank: (16 // tp + 4) * 2 for rank in range(tp * engines)}
    )


def test_partitioned_training_shards_still_form_one_complete_replica():
    training = [
        parameter("weight", rank, (rank * 2, 0), (2, 4), (4, 4)) for rank in range(2)
    ]
    train_meta = deserialize_value(merge_parameter_metadata(serialize_value(training)))
    inference = [parameter("weight", 0, (0, 0), (4, 4), (4, 4))]
    operations = TransferPlanBuilder(1, 2).build_weights_mapping_operations(
        inference, train_meta
    )
    assert len(train_meta[0].replicas) == 1
    assert {operation.send_rank for operation in operations} == {1, 2}
    assert sum(prod(operation.overlap_shape) for operation in operations) == 16
