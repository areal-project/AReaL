#!/usr/bin/env python3
"""Write a small n_per_source slice of a decision suite to JSONL.

Point ``--suite`` at any frozen suite directory (``manifest.json`` plus
``train.jsonl`` / ``development.jsonl``) or a folder of those JSONL files.

    python3 examples/decision/prepare_v7_sample.py \
        --suite /data/decision \
        --out-dir examples/decision/data \
        --n-per-source 32 --dev-per-source 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from areal.dataset.decision import (  # noqa: E402
    load_labelled_path,
    select_labelled_records,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_decision_sample(
    suite: Path,
    out_dir: Path,
    n_per_source: int = 32,
    dev_per_source: int = 4,
    verify_suite: bool = False,
) -> tuple[Path, Path]:
    train = select_labelled_records(
        load_labelled_path(suite, split="train", verify_suite=verify_suite),
        n_per_source=n_per_source,
    )
    dev = select_labelled_records(
        load_labelled_path(suite, split="development", verify_suite=verify_suite),
        n_per_source=dev_per_source,
    )
    train_path = out_dir / "poc-train.jsonl"
    dev_path = out_dir / "poc-dev.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    print(
        f"wrote {len(train)} train records -> {train_path}\n"
        f"wrote {len(dev)} development records -> {dev_path}"
    )
    return train_path, dev_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Decision suite directory or a folder containing train/development JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for the sampled JSONL files",
    )
    parser.add_argument("--n-per-source", type=int, default=32)
    parser.add_argument("--dev-per-source", type=int, default=4)
    parser.add_argument(
        "--verify-suite",
        action="store_true",
        help="Check suite checksums while sampling (slow, not needed for a PoC slice)",
    )
    args = parser.parse_args(argv)
    prepare_decision_sample(
        suite=args.suite,
        out_dir=args.out_dir,
        n_per_source=args.n_per_source,
        dev_per_source=args.dev_per_source,
        verify_suite=args.verify_suite,
    )


if __name__ == "__main__":
    main()
