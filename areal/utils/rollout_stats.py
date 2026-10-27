# SPDX-License-Identifier: Apache-2.0

"""Turn statistics transported with, and recorded from, retained rollout rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

from areal.utils.harness_metrics import (
    HARNESS_OUTCOME_IDS,
    classify_harness_outcome,
)
from areal.utils.harness_metrics import (
    HARNESS_OUTCOME_METRIC_CODES as HARNESS_OUTCOME_METRIC_CODES,
)

# One-dimensional row metadata: concat, DP redistribution, and row filtering
# must slice these exactly like rewards, not like token-aligned tensors.
ROLLOUT_TURN_COUNT = "rollout_num_turns"
ROLLOUT_HARNESS_OUTCOME = "rollout_harness_outcome"


def _count_generated_turns(
    trajectory: dict[str, torch.Tensor],
    interaction: InteractionWithTokenLogpReward | None,
) -> torch.Tensor:
    """Count generated turns at rollout export, independently of training masks."""
    turn_ids = trajectory["turn_ids"]
    if turn_ids.ndim != 2:
        raise ValueError("rollout turn_ids must have shape [batch, sequence]")
    if interaction is not None:
        if turn_ids.shape[0] != 1:
            raise ValueError("one rollout interaction must describe exactly one row")
        # Count the retained branch's real generation history, even when tensor
        # conversion fell back to a fresh turn ID after a non-growing prompt.
        # Sibling/retry branches are not ancestors and are never added here.
        turns = 0
        current = interaction
        while current is not None:
            if current.model_response is None:
                # Cached tensor-only interactions have no response history.
                # Fall back to the original exported IDs below.
                break
            turns += int(current.model_response.output_len > 0)
            current = current.parent
        else:
            return torch.tensor([turns], dtype=torch.float32, device=turn_ids.device)
    valid = turn_ids >= 0
    if "attention_mask" in trajectory:
        if trajectory["attention_mask"].shape != turn_ids.shape:
            raise ValueError("attention_mask must have the same shape as turn_ids")
        valid = valid & trajectory["attention_mask"].bool()
    # Nonnegative IDs identify generated tokens even when loss_mask is zero.
    # Exclude only prompts and original padding; later training edits must not
    # rewrite these observed rollout counts. Sorting tolerates gaps/repeated IDs.
    ordered_ids = torch.where(valid, turn_ids, -1).sort(dim=-1).values
    starts = ordered_ids >= 0
    starts[:, 1:] &= ordered_ids[:, 1:] != ordered_ids[:, :-1]
    return starts.sum(dim=-1).to(dtype=torch.float32)


def with_rollout_turn_stats(
    trajectory: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None,
    *,
    interaction: InteractionWithTokenLogpReward | None = None,
    require_harness_status: bool = False,
) -> dict[str, torch.Tensor]:
    """Snapshot compact statistics for one retained branch at tensor export.

    Counts describe the original rollout, not its trainable tokens. Preserve a
    snapshot already attached before HTTP transport erased the parent chain.
    Carry them
    unchanged through token masking/cropping; row selection/reordering must apply
    the same indices to these columns as to rewards and Harness outcomes.
    Read episode metadata here, not when the interaction's tensor cache was
    created: the final Harness status may be stamped after that cache exists.
    V2 requires an explicit status; missing metadata is unknown, not a confirmed
    non-Harness failure. This does not emit metrics for unconsumed rows.
    """
    if "turn_ids" not in trajectory:
        return trajectory
    if ROLLOUT_TURN_COUNT in trajectory:
        counts = trajectory[ROLLOUT_TURN_COUNT]
        if counts.shape != trajectory["turn_ids"].shape[:1]:
            raise ValueError("rollout turn counts must match the trajectory batch size")
    else:
        counts = _count_generated_turns(trajectory, interaction)

    outcome_id = classify_harness_outcome(
        metadata, require_status=require_harness_status
    )
    return {
        **trajectory,
        ROLLOUT_TURN_COUNT: counts,
        ROLLOUT_HARNESS_OUTCOME: torch.full_like(counts, outcome_id, dtype=torch.int32),
    }


def localize_rollout_turn_stats(
    batch: Sequence[dict[str, Any]],
) -> list[dict[str, torch.Tensor]]:
    """Fetch just row-aligned turn metadata, validating it before recording."""
    from areal.infra.rpc.rtensor import RTensor

    rows = []
    for trajectory in batch:
        if ROLLOUT_TURN_COUNT not in trajectory:
            continue
        row = {
            key: trajectory[key]
            for key in (ROLLOUT_TURN_COUNT, ROLLOUT_HARNESS_OUTCOME)
            if key in trajectory
        }
        rows.append(row)
    localized = RTensor.localize(rows)
    for trajectory, row in zip(
        (traj for traj in batch if ROLLOUT_TURN_COUNT in traj), localized, strict=True
    ):
        counts = row[ROLLOUT_TURN_COUNT]
        outcomes = row.get(ROLLOUT_HARNESS_OUTCOME)
        if counts.ndim != 1 or (
            outcomes is not None and outcomes.shape != counts.shape
        ):
            raise ValueError("rollout turn metadata must have one value per row")
        if outcomes is None:
            row[ROLLOUT_HARNESS_OUTCOME] = torch.full_like(
                counts, -1, dtype=torch.int32
            )
        input_ids = trajectory.get("input_ids")
        if input_ids is not None and counts.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "rollout turn metadata must match the trajectory batch size"
            )
    return localized


def record_rollout_turn_stats(
    batch: Sequence[dict[str, Any]], *, scope: str = "train"
) -> None:
    """Record turn distributions from the selected, filtered training rows.

    On a controller, fetch only the two small per-row columns, not the remote
    token tensors. In SPMD, each rank records its local DP shard; the existing
    actor stats export handles DP reduction without summing CP/TP/PP copies.
    """
    from areal.utils import stats_tracker
    from areal.utils.stats_tracker import ReduceType

    rows = localize_rollout_turn_stats(batch)
    # Other workflows need not provide turn metadata. An empty selected batch,
    # however, explicitly has zero samples (including post-filter rejection).
    if batch and not rows:
        return
    if rows:
        values = torch.cat([row[ROLLOUT_TURN_COUNT] for row in rows]).float()
        outcomes = torch.cat([row[ROLLOUT_HARNESS_OUTCOME] for row in rows])
        included = torch.ones_like(values, dtype=torch.bool)
    else:
        values = torch.zeros(1, dtype=torch.float32, device="cpu")
        outcomes = torch.zeros(1, dtype=torch.int32, device="cpu")
        included = torch.zeros(1, dtype=torch.bool, device="cpu")
    if values.numel() == 0:
        return

    masks = {
        "num_turns": included,
        "num_turns_no_harness_err": included & (outcomes == 0),
        "num_turns_harness_err": included & (outcomes > 0),
        **{
            f"num_turns_harness_err/{code}": included & (outcomes == outcome_id)
            for code, outcome_id in HARNESS_OUTCOME_IDS.items()
        },
    }
    tracker = stats_tracker.get(scope)
    # Evaluation records once per accepted task and exports after the full
    # validation window. Compact records avoid revalidating all previous tasks.
    ones = torch.ones_like(values)
    for metric, mask in masks.items():
        tracker.stat_compact(mask, ReduceType.AVG_MIN_MAX, **{metric: values})
        tracker.stat_compact(mask, ReduceType.SUM, **{f"{metric}_count": ones})
