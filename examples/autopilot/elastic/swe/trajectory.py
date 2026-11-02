# SPDX-License-Identifier: Apache-2.0

"""Adapt exported AReaL v2 trajectories to the SWE fixture's durable queue schema."""

import math
from typing import Any

_TOKEN_FIELDS = ("input_ids", "attention_mask", "loss_mask", "logprobs", "versions")


def trajectory_payload(
    tensors: dict[str, Any],
    *,
    task_id: str,
    rollout_id: str,
    model_revision: str,
    policy_version: int,
) -> dict[str, Any]:
    """Preserve original token IDs/logprobs; accept CPU tensors or nested lists.

    RTensor references must be localized while inference is alive. This payload
    contains values only, so later training does not depend on that process.
    """
    if not all(
        isinstance(value, str) and value
        for value in (task_id, rollout_id, model_revision)
    ):
        raise ValueError("Task, rollout and pinned model revision are required")
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("Invalid reconfigurator version")
    required = {*_TOKEN_FIELDS, "rewards"}
    if not required.issubset(tensors):
        raise ValueError(
            f"Missing trajectory fields: {sorted(required - tensors.keys())}"
        )
    values = {}
    for name in (*_TOKEN_FIELDS, "rewards", "original_rewards", "turn_ids"):
        if name not in tensors:
            continue
        value = tensors[name]
        if hasattr(value, "device"):
            if value.device.type != "cpu":
                raise ValueError("Queue serialization requires CPU tensors")
            value = value.tolist()
        if not isinstance(value, list):
            raise ValueError(
                f"Trajectory field {name} must contain materialized values"
            )
        values[name] = value
    rows = values["input_ids"]
    if not rows or not all(isinstance(row, list) and row for row in rows):
        raise ValueError("Expected nonempty [batch, sequence] token arrays")
    n_rows = len(rows)
    for name in (*_TOKEN_FIELDS, "turn_ids"):
        if name not in values:
            continue
        if len(values[name]) != n_rows or any(
            not isinstance(row, list) or len(row) != len(rows[index])
            for index, row in enumerate(values[name])
        ):
            raise ValueError(f"Trajectory shape mismatch: {name}")
    for name in ("rewards", "original_rewards"):
        if name in values and (
            len(values[name]) != n_rows
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values[name]
            )
        ):
            raise ValueError(f"Invalid trajectory rewards: {name}")
    generated = 0
    for index, ids in enumerate(rows):
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Token IDs must be nonnegative integers")
        masks, attention = values["loss_mask"][index], values["attention_mask"][index]
        versions, logprobs = values["versions"][index], values["logprobs"][index]
        for selected, valid, version, logprob in zip(
            masks, attention, versions, logprobs
        ):
            if (
                selected not in (0, 1)
                or valid not in (0, 1)
                or (selected and not valid)
            ):
                raise ValueError("Invalid loss/attention mask")
            if type(logprob) not in (int, float) or not math.isfinite(logprob):
                raise ValueError("Token logprobs must be finite numbers")
            if type(version) is not int or (selected and version != policy_version):
                raise ValueError("Generated token reconfigurator version mismatch")
            generated += int(selected)
    if not generated or any(not any(row) for row in values["loss_mask"]):
        raise ValueError("Each interaction must contain generated tokens")
    return {
        "schema_version": 1,
        "source": "swe",
        "task_id": task_id,
        "rollout_id": rollout_id,
        "model_revision": model_revision,
        "policy_version": policy_version,
        "generated_tokens": generated,
        "trajectory": values,
    }
