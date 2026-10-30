# SPDX-License-Identifier: Apache-2.0
import copy
import math
from collections import Counter

import pytest
import torch

pytest.importorskip("awex")
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.transfer.transfer_plan import TransferPlanBuilder

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.v2.weight_update.gateway.metadata import (
    merge_inference_metadata,
    merge_parameter_metadata,
)


def parameter(rank, world, *, name="weight", offset=0, size=16, total=16, tp_rank=None):
    shard = ParameterShardMeta(
        tp_rank=rank if tp_rank is None else tp_rank,
        attn_tp_rank=rank,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=rank,
        global_rank=rank,
        world_size=world,
        engine_rank=0,
        name=name,
        shape=(size,),
        numel=size,
        dtype=torch.float32,
        global_offset=(offset,),
        sharding_type=ShardingType.NO_SHARDING
        if size == total
        else ShardingType.TP_SHARDING,
        num_shards=1 if size == total else world,
        sharding_dim=0,
    )
    return ParameterMeta(
        name, total, (total,), torch.float32, [shard], [ParameterReplicaMeta([shard])]
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_transfer_covers_each_inference_shard_exactly_once(tp):
    train = [parameter(rank, 4) for rank in range(4)]
    infer = [
        parameter(rank, tp, offset=rank * (16 // tp), size=16 // tp)
        for rank in range(tp)
    ]
    train += [parameter(rank, 4, name="norm", size=4, total=4) for rank in range(4)]
    infer += [parameter(rank, tp, name="norm", size=4, total=4) for rank in range(tp)]
    raw = serialize_value(infer)
    untouched = copy.deepcopy(raw)
    normalized = merge_inference_metadata([{"meta": raw}, {"meta": copy.deepcopy(raw)}])
    assert raw == untouched
    normalized_train = deserialize_value(
        merge_parameter_metadata(serialize_value(train))
    )
    assert all(len(p.replicas) == 4 for p in normalized_train)
    builder = TransferPlanBuilder(
        infer_world_size=tp * 2, train_world_size=4, num_infer_engines=2
    )
    ops = builder.build_weights_mapping_operations(
        deserialize_value(normalized), normalized_train
    )
    volume, count = Counter(), Counter()
    for op in ops:
        key = op.recv_rank, op.recv_shard_meta.name
        volume[key] += math.prod(op.overlap_shape)
        count[key] += 1
    assert volume == Counter(
        {
            (rank, name): size
            for rank in range(tp * 2)
            for name, size in (("weight", 16 // tp), ("norm", 4))
        }
    )
    assert set(count.values()) == {1}


def test_fsdp_shards_with_same_tp_coordinate_form_one_complete_replica():
    shards = [
        parameter(rank, 2, offset=rank * 8, size=8, tp_rank=0) for rank in range(2)
    ]
    merged = deserialize_value(merge_parameter_metadata(serialize_value(shards)))
    assert len(merged) == 1 and len(merged[0].replicas) == 1
    assert [tuple(s.global_offset) for s in merged[0].replicas[0].shards] == [
        (0,),
        (8,),
    ]


@pytest.mark.parametrize("failure", ["missing", "overlap", "duplicate", "shape"])
def test_invalid_metadata_rejected_before_collective(failure):
    shards = [parameter(rank, 2, offset=rank * 8, size=8) for rank in range(2)]
    if failure == "missing":
        shards.pop()
    elif failure == "overlap":
        shards[1].shards[0].global_offset = (4,)
    elif failure == "duplicate":
        shards.append(copy.deepcopy(shards[0]))
    else:
        shards[1].global_shape = (17,)
    with pytest.raises(ValueError):
        merge_parameter_metadata(serialize_value(shards))


def test_inference_engine_layout_mismatch_rejected():
    first = serialize_value([parameter(0, 1)])
    second = serialize_value([parameter(0, 1, name="different")])
    with pytest.raises(ValueError, match="engines disagree"):
        merge_inference_metadata([{"meta": first}, {"meta": second}])


def test_empty_metadata_rejected():
    with pytest.raises(ValueError, match="Missing parameter metadata"):
        merge_parameter_metadata([])
