"""Summarize full-dataset, three-epoch GSM8K comparisons from trainer logs."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Epoch (\d+)/(\d+) Step (\d+)/(\d+) Train step (\d+)/(\d+) done\.")


def summarize(run: Path) -> dict:
    assert (run / "exit-code.txt").read_text().strip() == "0", run
    records = []
    record = None
    completed = False
    with (run / "grpo.log").open() as file:
        for raw_line in file:
            line = ANSI.sub("", raw_line)
            completed |= "Training completes!" in line
            assert "Traceback (most recent call last)" not in line
            match = STEP.search(line)
            if match:
                if record is not None:
                    records.append(record)
                epoch, n_epochs, epoch_step, steps_per_epoch, step, total_steps = map(
                    int, match.groups()
                )
                record = {
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "global_step": step,
                    "n_epochs": n_epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "metrics": {},
                }
            elif record is not None and line.startswith("│"):
                cells = [cell.strip() for cell in line.split("│")[1:-1]]
                for key, value in zip(cells[::2], cells[1::2]):
                    if key:
                        scalar = float(value)
                        assert math.isfinite(scalar), (run, step, key)
                        record["metrics"][key] = scalar
    if record is not None:
        records.append(record)

    assert completed and records, run
    first = records[0]
    assert first["n_epochs"] == 3
    assert len(records) == first["total_steps"]
    assert first["total_steps"] == 3 * first["steps_per_epoch"]
    assert [x["global_step"] for x in records] == list(
        range(1, first["total_steps"] + 1)
    )
    assert all(x["metrics"]["ppo_actor/update/update_successful"] == 1 for x in records)
    assert all(x["metrics"]["ppo_actor/update/grad_norm"] > 0 for x in records)

    epochs = []
    for epoch in range(1, 4):
        epoch_records = [x["metrics"] for x in records if x["epoch"] == epoch]
        assert len(epoch_records) == first["steps_per_epoch"]
        correct = sum(x["ppo_actor/correct_n_seqs"] for x in epoch_records)
        n_seqs = sum(x["ppo_actor/n_seqs"] for x in epoch_records)
        eval_rewards = [
            x["eval-rollout/reward"]
            for x in epoch_records
            if "eval-rollout/reward" in x
        ]
        assert len(eval_rewards) == 1, (run, epoch, eval_rewards)
        epochs.append(
            {
                "epoch": epoch,
                "updates": len(epoch_records),
                "training_correct": int(correct),
                "training_answers": int(n_seqs),
                "training_reward": correct / n_seqs,
                "evaluation_reward": eval_rewards[0],
                "mean_train_step_seconds": sum(
                    x["timeperf/train_step"] for x in epoch_records
                )
                / len(epoch_records),
                "min_grad_norm": min(
                    x["ppo_actor/update/grad_norm"] for x in epoch_records
                ),
                "max_grad_norm": max(
                    x["ppo_actor/update/grad_norm"] for x in epoch_records
                ),
                "last_lr": epoch_records[-1]["ppo_actor/update/lr"],
            }
        )

    peak_per_gpu = defaultdict(int)
    with (run / "gpu-memory.csv").open() as file:
        for row in csv.reader(file):
            if len(row) != 3:
                continue
            try:
                gpu = int(row[1].strip())
                used = int(row[2].strip())
            except ValueError:
                continue
            peak_per_gpu[gpu] = max(peak_per_gpu[gpu], used)
    start = datetime.fromisoformat(
        (run / "start.utc").read_text().strip().replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        (run / "end.utc").read_text().strip().replace("Z", "+00:00")
    )
    return {
        "updates": len(records),
        "steps_per_epoch": first["steps_per_epoch"],
        "epochs": epochs,
        "mean_train_step_seconds": sum(
            x["metrics"]["timeperf/train_step"] for x in records
        )
        / len(records),
        "wall_seconds": (end - start).total_seconds(),
        "peak_actor_gpu_memory_mib": max(peak_per_gpu[i] for i in range(4)),
        "peak_rollout_gpu_memory_mib": max(peak_per_gpu[i] for i in range(4, 8)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    report = {name: summarize(root / name) for name in ("adamw", "muon")}
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
