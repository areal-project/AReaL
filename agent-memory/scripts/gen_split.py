#!/usr/bin/env python3
"""One-shot script: generate configs/bigcodebench/splits/full_seed42.json"""
import json, random, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "bigcodebench" / "bigcodebench_full.jsonl"

problems = {}
with open(data_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        problems[str(task["task_id"])] = task

task_ids = sorted(problems.keys())
random.seed(42)
random.shuffle(task_ids)
split_idx = int(len(task_ids) * 0.7)
train_ids = task_ids[:split_idx]
val_ids = task_ids[split_idx:]

out = {
    "subset": "full",
    "split": "instruct",
    "seed": 42,
    "train_ratio": 0.7,
    "train_ids": train_ids,
    "val_ids": val_ids,
}
out_path = root / "configs" / "bigcodebench" / "splits" / "full_seed42.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"Total: {len(task_ids)}, Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"Saved: {out_path}")
