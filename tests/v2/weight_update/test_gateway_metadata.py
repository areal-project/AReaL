import pytest

pytest.importorskip("fastapi")

from areal.v2.weight_update.gateway.app import _merge_meta_by_name


def _serialized_shard(rank: int, offset: int, length: int) -> dict:
    return {
        "type": "dataclass",
        "class_path": "awex.meta.weight_meta.ParameterShardMeta",
        "data": {
            "name": "model.layers.0.mlp.gate_proj.weight",
            "global_rank": rank,
            "global_offset": [offset],
            "shape": [length],
            "dtype": {"type": "torch_dtype", "value": "torch.float32"},
            "sharding_type": {
                "type": "enum",
                "class_path": "awex.sharding.param_sharding.ShardingType",
                "value": "TP_SHARDING",
            },
            "sharding_dim": 0,
            "num_shards": 4,
        },
    }


def _serialized_param(shard: dict) -> dict:
    return {
        "type": "dataclass",
        "class_path": "awex.meta.weight_meta.ParameterMeta",
        "data": {
            "name": "model.layers.0.mlp.gate_proj.weight",
            "global_numel": 16,
            "global_shape": [16],
            "dtype": {"type": "torch_dtype", "value": "torch.float32"},
            "shards": [shard],
            "replicas": [
                {
                    "type": "dataclass",
                    "class_path": "awex.meta.weight_meta.ParameterReplicaMeta",
                    "data": {"shards": [shard]},
                }
            ],
        },
    }


def test_merge_inference_meta_dedupes_engines_and_keeps_tp_shards():
    # Two SGLang engines report the same per-engine TP metadata. The gateway
    # should keep one logical engine with all TP shards; TransferPlanBuilder
    # expands it across engines via num_infer_engines later.
    metas = []
    for _engine in range(2):
        for rank in range(4):
            metas.append(_serialized_param(_serialized_shard(rank, rank * 4, 4)))

    merged = _merge_meta_by_name(metas, "inference")

    assert len(merged) == 1
    data = merged[0]["data"]
    assert sorted(s["data"]["global_rank"] for s in data["shards"]) == [0, 1, 2, 3]
    assert len(data["shards"]) == 4
    replica_shards = data["replicas"][0]["data"]["shards"]
    assert sorted(s["data"]["global_rank"] for s in replica_shards) == [0, 1, 2, 3]
    assert len(replica_shards) == 4
