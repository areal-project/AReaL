"""
BigCodeBench task wrappers: data loading, train/val split, and sample output.

We expect the dataset to be stored as JSONL under:
  data/bigcodebench/bigcodebench_{subset}.jsonl

This repo does not ship the dataset by default. If the file is missing, we raise
with a concrete download command.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "bigcodebench"
DEFAULT_FULL_PATH = DEFAULT_DATA_DIR / "bigcodebench_full.jsonl"
DEFAULT_HARD_PATH = DEFAULT_DATA_DIR / "bigcodebench_hard.jsonl"


def load_bcb_data(subset: str = "hard", data_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load BigCodeBench dataset from a local JSONL file.

    Args:
        subset: "full" (1140 tasks) or "hard" (148 tasks)
        data_path: Optional explicit JSONL path. If provided, `subset` is ignored.
    """
    if data_path:
        path = Path(data_path)
    elif subset == "hard":
        path = DEFAULT_HARD_PATH
    elif subset == "full":
        path = DEFAULT_FULL_PATH
    else:
        raise ValueError(f"Unknown subset: {subset}. Use 'full' or 'hard'.")

    if not path.exists():
        # Keep the message actionable and mirror-friendly.
        raise FileNotFoundError(
            f"BigCodeBench dataset file not found: {path}\n"
            "Download (example, using datasets):\n"
            "  python - <<'PY'\n"
            "  from datasets import load_dataset\n"
            "  import json\n"
            f"  ds = load_dataset('bigcode/bigcodebench-{subset}', split='v0.1.4')\n"
            f"  path = r'{path}'\n"
            "  path_dir = __import__('pathlib').Path(path).parent\n"
            "  path_dir.mkdir(parents=True, exist_ok=True)\n"
            "  with open(path, 'w', encoding='utf-8') as f:\n"
            "    for item in ds:\n"
            "      f.write(json.dumps(item, ensure_ascii=False) + '\\n')\n"
            "  print('wrote', path)\n"
            "  PY\n"
        )

    problems: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            task_id = str(task["task_id"])
            problems[task_id] = task

    return problems


def split_dataset(
    problems: Dict[str, Dict[str, Any]],
    train_ratio: float = 0.7,
    seed: int = 42,
    split_file: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Split dataset into train and val sets.

    If `split_file` exists, uses it (expects JSON with `train_ids` and `val_ids`).
    """
    if split_file and os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_ids = list(data.get("train_ids") or [])
        val_ids = list(data.get("val_ids") or [])
        valid = set(problems.keys())
        train_ids = [tid for tid in train_ids if tid in valid]
        val_ids = [tid for tid in val_ids if tid in valid]
        return train_ids, val_ids

    task_ids = sorted(problems.keys())
    random.seed(seed)
    random.shuffle(task_ids)
    split_idx = int(len(task_ids) * float(train_ratio))
    train_ids = task_ids[:split_idx]
    val_ids = task_ids[split_idx:]
    return train_ids, val_ids


def get_prompt(task: Dict[str, Any], split: str = "instruct") -> str:
    """Get the prompt for a BCB task."""
    if split == "instruct":
        return str(task["instruct_prompt"])
    if split == "complete":
        return str(task["complete_prompt"])
    raise ValueError(f"Unknown split: {split}. Use 'instruct' or 'complete'.")


def timestamp_dir(root: str, name: str) -> str:
    """Create a timestamped output directory and return its path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = (name or "model").replace("/", "_")
    out = os.path.join(root, f"{ts}_{safe}")
    os.makedirs(out, exist_ok=True)
    return out


def write_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Write samples to JSONL (one per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

