# SPDX-License-Identifier: Apache-2.0
"""Compare uninterrupted and recovered next-step MTP state and losses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml


def _load_record(path: Path, global_step: int) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [record for record in records if record["global_step"] == global_step]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one global_step={global_step} record in {path}, got {len(matches)}"
        )
    return matches[0]["stats"]


def _find_metric(stats: dict, suffix: str) -> float:
    matches = [(key, value) for key, value in stats.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"expected one *{suffix} metric, got {matches}")
    return float(matches[0][1])


def _diff(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.float()
    actual = actual.float()
    absolute = (reference - actual).abs()
    relative = absolute / reference.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return float(absolute.max()), float(relative.max())


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    tolerances = yaml.safe_load(
        (script_dir / "acceptance_tolerances.yaml").read_text()
    )["comparisons"]["continuous_vs_recovered_next_step"]
    snapshot_root = Path(os.environ["SNAPSHOT_ROOT"])
    continuous = sorted(
        snapshot_root.glob(
            os.environ.get("CONTINUOUS_SNAPSHOT_LABEL", "continuous-step5")
            + ".rank-*.pt"
        )
    )
    recovered = sorted(
        snapshot_root.glob(
            os.environ.get("RECOVERED_SNAPSHOT_LABEL", "recovered-step5") + ".rank-*.pt"
        )
    )
    if len(continuous) != 16 or len(recovered) != 16:
        raise AssertionError(
            f"expected 16 snapshots per run, got {len(continuous)} and {len(recovered)}"
        )

    max_abs = 0.0
    max_rel = 0.0
    compared = 0
    for continuous_path, recovered_path in zip(continuous, recovered, strict=True):
        lhs = torch.load(continuous_path, map_location="cpu", weights_only=True)
        rhs = torch.load(recovered_path, map_location="cpu", weights_only=True)
        if lhs["rank"] != rhs["rank"]:
            raise AssertionError(f"rank mismatch: {lhs['rank']} != {rhs['rank']}")
        if lhs["input_digest"] != rhs["input_digest"]:
            raise AssertionError(
                f"Next-update input batch differs on rank {lhs['rank']}"
            )
        lhs_parameters = lhs["parameters"]
        rhs_parameters = rhs["parameters"]
        if lhs_parameters.keys() != rhs_parameters.keys():
            raise AssertionError(
                f"parameter-key mismatch on rank {lhs['rank']}: "
                f"missing={sorted(lhs_parameters.keys() - rhs_parameters.keys())}, "
                f"extra={sorted(rhs_parameters.keys() - lhs_parameters.keys())}"
            )
        for name, reference in lhs_parameters.items():
            actual = rhs_parameters[name]
            torch.testing.assert_close(
                actual,
                reference,
                atol=tolerances["parameter_atol"],
                rtol=tolerances["parameter_rtol"],
                msg=lambda message: f"rank {lhs['rank']} parameter {name}: {message}",
            )
            abs_diff, rel_diff = _diff(reference, actual)
            max_abs = max(max_abs, abs_diff)
            max_rel = max(max_rel, rel_diff)
            compared += 1

    continuous_stats = _load_record(
        Path(os.environ["CONTINUOUS_METRICS_FILE"]),
        int(os.environ.get("COMPARE_GLOBAL_STEP", "4")),
    )
    recovered_stats = _load_record(
        Path(os.environ["RECOVERED_METRICS_FILE"]),
        int(os.environ.get("COMPARE_GLOBAL_STEP", "4")),
    )
    loss = _find_metric(continuous_stats, "/loss/avg")
    recovered_loss = _find_metric(recovered_stats, "/loss/avg")
    mtp_loss = _find_metric(continuous_stats, "/mtp_loss")
    recovered_mtp_loss = _find_metric(recovered_stats, "/mtp_loss")
    loss_abs = abs(loss - recovered_loss)
    loss_rel = loss_abs / max(abs(loss), torch.finfo(torch.float32).tiny)
    mtp_loss_abs = abs(mtp_loss - recovered_mtp_loss)
    mtp_loss_rel = mtp_loss_abs / max(abs(mtp_loss), torch.finfo(torch.float32).tiny)

    assert loss_abs <= tolerances["loss_atol"] + tolerances["loss_rtol"] * abs(loss)
    assert mtp_loss_abs <= tolerances["mtp_loss_atol"] + tolerances[
        "mtp_loss_rtol"
    ] * abs(mtp_loss)
    from examples.mtp.qwen38_flash_next.recovery_checks import assert_state_close

    state_root = Path(os.environ["RECOVERY_STATE_DIR"])
    boundary_tensors = 0
    optimizer_update_tensors = 0
    for rank in range(16):

        def load(label):
            return torch.load(
                state_root / f"{label}.rank-{rank:02d}.pt",
                map_location="cpu",
                weights_only=False,
            )

        saved = load("saved-boundary")
        restored = load("restored-boundary")
        boundary_tensors += assert_state_close(saved, restored, atol=0.0, rtol=0.0)
        initial = load(
            os.environ.get("CONTINUOUS_SNAPSHOT_LABEL", "continuous-step5") + "-initial"
        )["frozen_parameters"]
        recovered_initial = load(
            os.environ.get("RECOVERED_SNAPSHOT_LABEL", "recovered-step5") + "-initial"
        )["frozen_parameters"]
        for frozen in (
            saved["frozen_parameters"],
            restored["frozen_parameters"],
            recovered_initial,
        ):
            assert_state_close(
                initial, frozen, atol=0.0, rtol=0.0, path="frozen_parameters"
            )
        lhs = load(
            os.environ.get("CONTINUOUS_SNAPSHOT_LABEL", "continuous-step5") + "-state"
        )
        rhs = load(
            os.environ.get("RECOVERED_SNAPSHOT_LABEL", "recovered-step5") + "-state"
        )
        for state in (lhs, rhs):
            assert_state_close(
                initial,
                state["frozen_parameters"],
                atol=0.0,
                rtol=0.0,
                path="frozen_parameters",
            )
        optimizer_update_tensors += assert_state_close(
            lhs["optimizer"],
            rhs["optimizer"],
            atol=tolerances["optimizer_state_atol"],
            rtol=tolerances["optimizer_state_rtol"],
        )
        assert_state_close(lhs["scheduler"], rhs["scheduler"], atol=0.0, rtol=0.0)

    result = {
        "declared_tolerances": tolerances,
        "boundary_state_exact_tensors": boundary_tensors,
        "next_update_optimizer_tensors": optimizer_update_tensors,
        "parameter_tensors_compared": compared,
        "parameter_max_abs_diff": max_abs,
        "parameter_max_rel_diff": max_rel,
        "loss_reference": loss,
        "loss_recovered": recovered_loss,
        "loss_max_abs_diff": loss_abs,
        "loss_max_rel_diff": loss_rel,
        "mtp_loss_reference": mtp_loss,
        "mtp_loss_recovered": recovered_mtp_loss,
        "mtp_loss_max_abs_diff": mtp_loss_abs,
        "mtp_loss_max_rel_diff": mtp_loss_rel,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("compare_recovery PASS")


if __name__ == "__main__":
    main()
