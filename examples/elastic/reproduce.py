# SPDX-License-Identifier: Apache-2.0

"""Run the fixed eight-step recipe through the existing inline demo entry point."""

import argparse
import datetime
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def check_hash(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Reproduction input differs: {path.name}")


def check_inputs(model: Path, dataset: Path, lock: dict, *, full_model: bool) -> None:
    check_hash(dataset, lock["dataset"]["tasks_sha256"])
    check_hash(dataset.with_suffix(".images.json"), lock["dataset"]["images_sha256"])
    for name, expected in lock["model"]["sha256"].items():
        path = model / name
        if not path.is_file():
            raise ValueError(f"Incomplete model: {name}")
        if full_model or not name.endswith(".safetensors"):
            check_hash(path, expected)


def build_command(args: argparse.Namespace, lock: dict) -> list[str]:
    command = [sys.executable, "-u", "-m", "examples.elastic.run_demo"]
    for key in ("model", "dataset", "swe_runtime", "output", "policy"):
        value = getattr(args, key)
        command += ["--" + key.replace("_", "-"), str(value)]
    for key, value in lock["recipe"].items():
        command += ["--" + key, str(value)]
    if args.prepare_only:
        command.append("--prepare-only")
    return command


def run_command(command: list[str]) -> int:
    """Forward one stop request, then let run_demo finish resource cleanup."""
    process = subprocess.Popen(command, start_new_session=True)
    forwarded = False

    def forward(signum, frame):
        nonlocal forwarded
        if not forwarded and process.poll() is None:
            forwarded = True
            process.send_signal(signum)

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("resident", "agent"), required=True)
    for name in ("model", "dataset", "swe-runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--verify-model", action="store_true", help="Also hash all model weight shards"
    )
    args = parser.parse_args()
    for key in ("model", "dataset", "swe_runtime", "output"):
        setattr(args, key, getattr(args, key).resolve())
    recipe_dir = Path(__file__).parent
    lock = json.loads((recipe_dir / "reproduction/inputs.lock.json").read_text())
    try:
        if args.output.exists():
            raise ValueError("Use a new output directory")
        check_hash(recipe_dir / "config.yaml", lock["config_sha256"])
        check_inputs(args.model, args.dataset, lock, full_model=args.verify_model)
        if not args.prepare_only:
            parent = args.output.parent
            while not parent.exists():
                parent = parent.parent
            required = lock["required_free_gib"][args.policy]
            if shutil.disk_usage(parent).free < required * 1024**3:
                raise ValueError(
                    f"Need at least {required} GiB free on output filesystem"
                )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    start = time.monotonic()
    returncode = run_command(build_command(args, lock))
    elapsed = time.monotonic() - start
    if args.output.is_dir():
        (args.output / "invocation.json").write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "elapsed_seconds": elapsed,
                    "exit_code": returncode,
                    "prepare_only": args.prepare_only,
                    "model_weights_hash_checked": args.verify_model,
                    "timing_scope": "run_demo subprocess, including setup and cleanup; excludes reproduction input checks",
                },
                indent=2,
            )
            + "\n"
        )
    raise SystemExit(returncode if returncode >= 0 else 128 - returncode)


if __name__ == "__main__":
    main()
