#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json_dict(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object (dict) in {path}, got {type(obj)}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge LLB train/val JSON splits into a single dict dataset (for train-only runs)."
    )
    ap.add_argument("--train", required=True, type=Path, help="Path to *_train.json")
    ap.add_argument("--val", required=True, type=Path, help="Path to *_val.json")
    ap.add_argument("--out", required=True, type=Path, help="Output path for merged JSON")
    ap.add_argument(
        "--indent", type=int, default=2, help="JSON indent for output (default: 2)"
    )
    args = ap.parse_args()

    train = _load_json_dict(args.train)
    val = _load_json_dict(args.val)

    overlap = set(train).intersection(val)
    if overlap:
        ex = next(iter(overlap))
        raise ValueError(
            f"Train/val keys overlap ({len(overlap)} overlaps). Example key: {ex}"
        )

    merged = dict(train)
    merged.update(val)  # append val keys after train keys (stable order)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(merged, ensure_ascii=False, indent=args.indent) + "\n",
        encoding="utf-8",
    )

    print(f"train: {args.train} -> {len(train)}")
    print(f"val:   {args.val} -> {len(val)}")
    print(f"out:   {args.out} -> {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

