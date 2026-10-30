# SPDX-License-Identifier: Apache-2.0

"""Prepare SWE-Gym training tasks or existing SWE fixtures, without gold answers."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def write_dataset(
    paths: list[Path],
    output: Path,
    *,
    repeats: int,
    seed: int,
    source: dict,
    images: list[dict] | None = None,
) -> dict:
    if repeats < 1 or output.exists():
        raise ValueError("repeats must be positive and output must be new")
    fixtures = []
    manifest = []
    seen = set()
    for path in sorted(paths):
        row = json.loads(path.read_text())
        for key in ("instance_id", "problem_statement", "eval_script", "base_commit"):
            if not row.get(key):
                raise ValueError(f"Fixture {path.name} is missing {key}")
        if row["instance_id"] in seen:
            raise ValueError(f"Duplicate task fixture: {row['instance_id']}")
        seen.add(row["instance_id"])
        row.pop("patch", None)
        row.pop("hints_text", None)
        fixtures.append(row)
        manifest.append(
            {
                "instance_id": row["instance_id"],
                "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_image": row.get("source_image"),
                "sandbox_image": row.get("sandbox_image", row["instance_id"]),
                "provenance": row.get("provenance"),
            }
        )
    if not fixtures:
        raise ValueError("No fixtures found")
    resolved_images = None
    if images is not None:
        by_task = {image["task"]: image for image in images}
        if len(by_task) != len(images) or not seen.issubset(by_task):
            raise ValueError(
                "Resolved images must cover each selected task exactly once"
            )
        resolved_images = [by_task[task] for task in sorted(seen)]
        if any(
            not image.get("image")
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("digest", ""))
            for image in resolved_images
        ):
            raise ValueError(
                "Each resolved task image requires an immutable sha256 digest"
            )
    rng = random.Random(seed)
    rows = []
    for _ in range(repeats):
        epoch = fixtures.copy()
        rng.shuffle(epoch)
        rows.extend(epoch)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = {
        "source": source,
        "tasks": manifest,
        "seed": seed,
        "repeats": repeats,
        "rows": len(rows),
        "unique_tasks": len(fixtures),
        "tasks_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    if resolved_images is not None:
        image_path = output.with_suffix(".images.json")
        image_path.write_text(json.dumps(resolved_images, indent=2) + "\n")
        result["images"] = {
            "file": image_path.name,
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        }
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixtures", type=Path)
    inputs.add_argument("--gym-parquet", type=Path)
    parser.add_argument(
        "--harness", type=Path, help="Pinned SWE-Gym/SWE-Bench-Fork checkout"
    )
    parser.add_argument("--dataset-revision")
    parser.add_argument("--harness-revision")
    parser.add_argument(
        "--images",
        type=Path,
        help="Resolved task/image/digest JSON list from image provisioning",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("repeats must be positive and output must be new")
    source = {"kind": "local fixtures", "dataset": "see per-task provenance"}
    if args.gym_parquet:
        import pyarrow.parquet as pq

        from .gym import build_gym_fixture

        if not (args.harness and args.dataset_revision and args.harness_revision):
            parser.error(
                "SWE-Gym requires --harness, --dataset-revision and --harness-revision"
            )
        rows = sorted(
            pq.read_table(args.gym_parquet).to_pylist(),
            key=lambda row: row["instance_id"],
        )
        if not 2 <= args.count <= len(rows):
            parser.error("count must select at least two distinct SWE-Gym tasks")
        source = {
            "dataset": "SWE-Gym/SWE-Gym",
            "split": "train",
            "revision": args.dataset_revision,
            "parquet_sha256": hashlib.sha256(args.gym_parquet.read_bytes()).hexdigest(),
            "harness": "SWE-Gym/SWE-Bench-Fork",
            "harness_revision": args.harness_revision,
            "population": len(rows),
            "selection": "seeded uniform sample of instance_id-sorted training split; no reward or latency filter",
        }
        args.fixtures = args.output.with_suffix(".fixtures")
        args.fixtures.mkdir(parents=True, exist_ok=False)
        for row in random.Random(args.seed).sample(rows, args.count):
            fixture = build_gym_fixture(row, args.harness, provenance=source)
            (args.fixtures / (row["instance_id"] + ".json")).write_text(
                json.dumps(fixture) + "\n"
            )
    write_dataset(
        list(args.fixtures.glob("*.json")),
        args.output,
        repeats=args.repeats,
        seed=args.seed,
        source=source,
        images=json.loads(args.images.read_text()) if args.images else None,
    )


if __name__ == "__main__":
    main()
