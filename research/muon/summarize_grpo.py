"""Extract auditable metrics from an actual completed GRPO output directory."""

import argparse
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    log = (args.output / "grpo.log").read_text()
    steps = {}
    current = None
    for line in log.splitlines():
        match = re.search(r"Train step (\d+)/\d+ done\.", line)
        if match:
            current = int(match.group(1))
            steps[current] = {}
        if current is None or not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        for key, value in zip(cells[::2], cells[1::2]):
            if key:
                scalar = float(value)
                assert math.isfinite(scalar), (current, key, value)
                steps[current][key] = scalar
    assert len(steps) >= 3, steps.keys()
    assert "Training completes!" in log
    assert "Traceback (most recent call last)" not in log
    for metrics in steps.values():
        assert metrics["ppo_actor/update/update_successful"] == 1
        for key in ("actor_loss/avg", "grad_norm", "lr"):
            assert math.isfinite(metrics[f"ppo_actor/update/{key}"])
        assert metrics["timeperf/update_weights"] > 0
    assert any("eval-rollout/reward" in metrics for metrics in steps.values())

    rollouts = []
    for path in sorted((args.output / "grpo").rglob("*.jsonl")):
        with path.open() as file:
            for line in file:
                record = json.loads(line)
                assert math.isfinite(record["reward"])
                rollouts.append(
                    {
                        "file": str(path.relative_to(args.output)),
                        **{
                            key: record[key]
                            for key in (
                                "task_id",
                                "sample_idx",
                                "seqlen",
                                "prompt_len",
                                "head_version",
                                "tail_version",
                                "reward",
                            )
                        },
                    }
                )
    assert rollouts
    evaluation = [r for r in rollouts if "/eval-rollout/" in r["file"]]
    assert evaluation and all(r["head_version"] == 3 for r in evaluation)
    pointers = list((args.output / "grpo").rglob("LATEST"))
    assert pointers, "Recovery checkpoint was not published"
    report = {
        "source_output": str(args.output),
        "steps": steps,
        "rollouts": rollouts,
        "recovery_pointers": [str(p.relative_to(args.output)) for p in pointers],
        "all_logged_scalars_finite": True,
        "note": "Scalars retain the precision printed by StatsLogger. Process exit status is recorded separately by RSI.",
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
