# SPDX-License-Identifier: Apache-2.0

"""Run a data recipe with python3 -m areal.v2.data_engineering."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import DataJobConfig, export_hf_dataset, run_data_job


def main(argv: list[str] | None = None) -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Execute a data recipe and publish a verified task dataset."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON or YAML data job configuration"
    )
    parser.add_argument("--output", type=Path, required=True, help="New job directory")
    parser.add_argument(
        "--export-hf",
        type=Path,
        help="Also materialize the training view in a new HuggingFace dataset directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing job with identical config and sources",
    )
    args = parser.parse_args(argv)
    values = yaml.safe_load(args.config.read_text())
    if not isinstance(values, dict):
        parser.error("Data config must be an object")
    config = DataJobConfig.from_dict(values, base_dir=args.config.resolve().parent)
    snapshot = asyncio.run(run_data_job(config, args.output, resume=args.resume))
    result = {"dataset_id": snapshot.dataset_id, "dataset_path": str(snapshot.root)}
    if args.export_hf is not None:
        result["training_dataset_path"] = str(
            export_hf_dataset(snapshot, args.export_hf)
        )
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
