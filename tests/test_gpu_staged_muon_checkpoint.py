# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.megatron_utils.gpu_staged_muon import merge_muon_checkpoint_metadata


def _metadata(rank: int) -> dict:
    groups = {}
    for name in ("dp", "dp_cp", "tp", "cp", "ep", "expt_tp", "expt_dp", "pp"):
        members = [0, 1] if name in {"dp", "dp_cp", "expt_dp"} else [rank]
        groups[name] = {
            "size": len(members),
            "rank": members.index(rank),
            "members": members,
        }
    parameters = []
    for index, name in enumerate(
        (f"norm_{rank}", "embedding") if rank == 0 else (f"norm_{rank}",)
    ):
        parameters.append(
            {
                "name": name,
                "domain": "dense",
                "coordinate": {"pp": 0, "tp": 0},
                "shape": [2],
                "dtype": "torch.bfloat16",
                "state_kinds": ["master_param", "exp_avg", "exp_avg_sq"],
                "source_owner": {
                    "global_rank": rank,
                    "owner_rank": rank,
                    "owner_ordinal": index,
                    "group_index": index,
                    "parameter_index": 0,
                    "unit_order": index,
                },
            }
        )
    return {
        "schema_version": 2,
        "megatron_core_version": "0.17.0",
        "emerging_optimizers_version": "0.3.0",
        "topology": {"world_size": 2, "global_rank": rank, "groups": groups},
        "algorithm": {},
        "leaf_tree": [
            {
                "tree_path": [0],
                "kind": "scalar_adamw",
                "parameters": parameters,
                "param_groups": [
                    {"lr": 0.02, "weight_decay": 0.0, "step": 1},
                    {"lr": 0.02, "weight_decay": 0.1, "step": 1 - rank},
                ],
            }
        ],
    }


@pytest.mark.parametrize("ranks", [[0, 1], [1, 0]])
def test_checkpoint_partially_empty_groups_select_nonempty_owner(
    ranks: list[int],
) -> None:
    """Empty-owner step counters cannot override authoritative optimizer state."""
    metadata = [_metadata(rank) for rank in ranks]
    merged = merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=ranks)
    groups = merged["leaf_tree"][0]["param_groups"]
    assert [group["step"] for group in groups] == [1, 1]
    assert len(merged["leaf_tree"][0]["parameters"]) == 3
    assert metadata[ranks.index(1)]["leaf_tree"][0]["param_groups"][1]["step"] == 0


@pytest.mark.parametrize("field,value", [("step", 2), ("lr", 0.03)])
def test_checkpoint_nonempty_owners_conflict_raises(field: str, value: float) -> None:
    """Real owners must still agree on counters and optimizer settings."""
    metadata = [_metadata(0), _metadata(1)]
    metadata[1]["leaf_tree"][0]["param_groups"][0][field] = value
    with pytest.raises(ValueError, match="conflicts across owners"):
        merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=[0, 1])


@pytest.mark.parametrize("conflicting", [False, True])
def test_checkpoint_globally_empty_group_preserves_validation(
    conflicting: bool,
) -> None:
    """Groups with no owner retain their metadata and reject inconsistencies."""
    metadata = [_metadata(0), _metadata(1)]
    for rank, item in enumerate(metadata):
        item["leaf_tree"][0]["param_groups"].append(
            {"lr": 0.02, "step": int(conflicting and rank == 1)}
        )
    if conflicting:
        with pytest.raises(ValueError, match="conflicts across owners"):
            merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=[0, 1])
    else:
        merged = merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=[0, 1])
        assert merged["leaf_tree"][0]["param_groups"][2] == {"lr": 0.02, "step": 0}


def test_checkpoint_invalid_owner_group_index_raises() -> None:
    """A parameter cannot claim ownership of a nonexistent group."""
    metadata = [_metadata(0), _metadata(1)]
    metadata[0]["leaf_tree"][0]["parameters"][0]["source_owner"]["group_index"] = 2
    with pytest.raises(ValueError, match="group_index is out of range"):
        merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=[0, 1])


def test_checkpoint_nonempty_group_count_mismatch_raises() -> None:
    """Partially populated leaves must agree on the group structure."""
    metadata = [_metadata(0), _metadata(1)]
    metadata[1]["leaf_tree"][0]["param_groups"].pop()
    with pytest.raises(ValueError, match="param-group count conflicts"):
        merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=[0, 1])
