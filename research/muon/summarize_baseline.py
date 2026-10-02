"""Summarize the three-epoch GSM8K runs launched from the stock Megatron YAML."""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def summarize(root: Path, optimizer: str) -> dict:
    run = root / optimizer
    assert (run / "exit-code.txt").read_text().strip() == "0", optimizer
    log = ANSI.sub("", (run / "grpo.log").read_text())
    assert "Training completes!" in log
    assert "Traceback (most recent call last)" not in log
    steps: dict[int, dict[str, float]] = {}
    current = None
    for line in log.splitlines():
        match = re.search(r"Epoch (\d)/3 Step 1/1 Train step (\d)/3 done\.", line)
        if match:
            current = int(match.group(2))
            assert int(match.group(1)) == current
            steps[current] = {}
        if current is None or not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        for key, value in zip(cells[::2], cells[1::2]):
            if key:
                scalar = float(value)
                assert math.isfinite(scalar), (optimizer, current, key)
                steps[current][key] = scalar
    assert sorted(steps) == [1, 2, 3], (optimizer, sorted(steps))
    assert all(
        step["ppo_actor/update/update_successful"] == 1 for step in steps.values()
    )
    assert all(step["timeperf/update_weights"] > 0 for step in steps.values())
    assert all("eval-rollout/reward" in step for step in steps.values())

    peak_per_gpu: dict[int, int] = {}
    with (run / "gpu-memory.csv").open() as file:
        for row in csv.reader(file):
            if len(row) != 3:
                continue
            try:
                gpu = int(row[1].strip())
                used = int(row[2].strip())
            except ValueError:
                continue
            peak_per_gpu[gpu] = max(peak_per_gpu.get(gpu, 0), used)
    start = datetime.fromisoformat(
        (run / "start.utc").read_text().strip().replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        (run / "end.utc").read_text().strip().replace("Z", "+00:00")
    )
    return {
        "optimizer": optimizer,
        "updates": 3,
        "nonzero_gradient_steps": sum(
            step["ppo_actor/update/grad_norm"] > 0 for step in steps.values()
        ),
        "task_reward_by_epoch": [
            steps[i]["ppo_actor/task_reward/avg"] for i in (1, 2, 3)
        ],
        "evaluation_reward_by_epoch": [
            steps[i]["eval-rollout/reward"] for i in (1, 2, 3)
        ],
        "train_step_seconds_by_epoch": [
            steps[i]["timeperf/train_step"] for i in (1, 2, 3)
        ],
        "mean_train_step_seconds": sum(
            steps[i]["timeperf/train_step"] for i in (1, 2, 3)
        )
        / 3,
        "wall_seconds": (end - start).total_seconds(),
        "peak_actor_gpu_memory_mib": max(peak_per_gpu[i] for i in range(4)),
        "peak_rollout_gpu_memory_mib": max(peak_per_gpu[i] for i in range(4, 8)),
        "peak_per_gpu_memory_mib": peak_per_gpu,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    report = {name: summarize(root, name) for name in ("adamw", "muon")}
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
