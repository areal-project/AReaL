# SPDX-License-Identifier: Apache-2.0

"""Compare same-version FSDP checkpoints one global DCP leaf at a time on CPU."""

import hashlib
import json
from pathlib import Path


def dcp_fingerprints(checkpoint: Path) -> dict:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    path = checkpoint / "dcp"
    metadata = dcp.FileSystemReader(path).read_metadata()
    result = {}
    for name, item in sorted(metadata.state_dict_metadata.items()):
        tensor = isinstance(item, TensorStorageMetadata)
        state = {
            name: torch.empty(item.size, dtype=item.properties.dtype, device="cpu")
            if tensor
            else None
        }
        dcp.load(
            state,
            checkpoint_id=path,
            no_dist=True,
            planner=dcp.DefaultLoadPlanner(
                flatten_state_dict=False, flatten_sharded_tensors=False
            ),
        )
        value = state[name]
        if tensor:
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            result[name] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
            }
            del raw
        else:
            # Preserve Python types as well as values (e.g. tuple/list param groups).
            result[name] = {
                "kind": "value",
                "type": type(value).__name__,
                "value": value,
            }
        result[name]["logical_path"] = metadata.planner_data[name]
        del state, value
    return result


def compare_dcp(source: Path, restored: Path) -> dict:
    left, right = dcp_fingerprints(source), dcp_fingerprints(restored)
    if left != right:
        different = sorted(
            name
            for name in left.keys() | right.keys()
            if left.get(name) != right.get(name)
        )
        raise ValueError(f"Restored model/optimizer DCP differs: {different[:10]}")
    if not any(name.startswith("dcp.model.") for name in left) or not any(
        name.startswith("dcp.optim.") for name in left
    ):
        raise ValueError("Audit requires both model and optimizer state")
    return {
        "status": "exact",
        "tensor_leaves": sum(row["kind"] == "tensor" for row in left.values()),
        "value_leaves": sum(row["kind"] == "value" for row in left.values()),
        "fingerprints": left,
    }


def compare_runtime(source: Path, restored: Path, restore_reports: list[dict]) -> dict:
    from areal.utils.atomic import recovery_digest

    original = [
        json.loads(path.read_text()) for path in sorted(source.glob("runtime-*.json"))
    ]
    actual = [
        json.loads(path.read_text()) for path in sorted(restored.glob("runtime-*.json"))
    ]
    reports = {row["rank"]: row for row in restore_reports}
    if (
        not original
        or not actual
        or len(actual) != actual[0]["world_size"]
        or set(reports) != set(range(len(actual)))
    ):
        raise ValueError(
            "Runtime audit requires every restored rank and its load report"
        )
    for row in actual:
        if (
            row["world_size"] != len(actual)
            or recovery_digest(row) != reports[row["rank"]]["state_sha256"]
        ):
            raise ValueError(
                "Restored rank RNG/scheduler state differs from its approved load state"
            )
        for key in (
            "identity_sha256",
            "policy_version",
            "backend",
            "lr_scheduler",
            "optimizer_group_lrs",
        ):
            if any(item[key] != row[key] for item in original):
                raise ValueError(f"Runtime continuity differs: {key}")
    return {
        "status": "exact",
        "source_world_size": len(original),
        "target_world_size": len(actual),
        "policy_version": actual[0]["policy_version"],
        "rng": "matches approved target-effective load reports",
    }
