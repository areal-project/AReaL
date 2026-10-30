# SPDX-License-Identifier: Apache-2.0

"""Archive metadata and retire full checkpoints of a completed, closed demo run."""

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

from areal.utils import logging
from areal.utils.atomic import atomic_json


def retire_checkpoint(run: Path) -> dict:
    run = run.resolve()
    result = json.loads((run / "result.json").read_text())
    status = json.loads((run / "state/status.json").read_text())
    if (
        result.get("status") != "training_completed"
        or not status.get("coordinator_closed")
        or status.get("actor_running")
    ):
        raise ValueError(
            "Only a completed run with a closed coordinator and stopped Actor can be retired"
        )
    record_path = run / "checkpoint-retired.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    checkpoint = run / "state/checkpoints"
    if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(run):
        raise ValueError("Checkpoint directory must be inside this run")
    if not (checkpoint / "latest.json").is_file():
        raise ValueError("No committed checkpoint to retire")
    files = [p for p in checkpoint.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in checkpoint.rglob("*")):
        raise ValueError("Checkpoint symlinks are unsupported")
    archive = run / "checkpoint-metadata.tar.gz"
    if archive.exists():
        raise ValueError("Metadata archive already exists; inspect it before retrying")
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as stream:
        for path in files:
            if path.suffix != ".distcp" and path.stat().st_size <= 10 * 1024**2:
                stream.add(path, arcname=str(path.relative_to(checkpoint)))
    with tarfile.open(temporary) as stream:
        names = stream.getnames()
        if "latest.json" not in names or not any(
            name.endswith("/manifest.json") for name in names
        ):
            raise ValueError("Checkpoint metadata archive is incomplete")
    temporary.replace(archive)
    record = {
        "bytes_removed": sum(path.stat().st_size for path in files),
        "metadata_archive": archive.name,
        "full_resume_available": False,
        "reason": "Completed demo; preserve evidence and reclaim full checkpoint storage",
        "wall_time": time.time(),
    }
    shutil.rmtree(checkpoint)
    atomic_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    result = retire_checkpoint(parser.parse_args().run)
    logging.getLogger("RLTrainer").info(
        "Retired %.2f GiB of checkpoint files; metadata retained, full-state resume unavailable",
        result["bytes_removed"] / 1024**3,
    )


if __name__ == "__main__":
    main()
