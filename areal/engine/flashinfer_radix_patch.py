# SPDX-License-Identifier: Apache-2.0
"""Prepare a private FlashInfer runtime with the upstream radix barrier fix.

The leading CTA must not reset a barrier while peers still observe it. This
backport uses last-CTA reset from FlashInfer 49f2abf and distinct JIT module names
so previously compiled sampling/top-k binaries cannot bypass the fixed header.
Run preparation inside the inference image, then prepend the output directory
to the inference workers' PYTHONPATH. No installed package is modified.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path


def _specification() -> dict:
    return json.loads(
        Path(__file__).with_name("flashinfer_radix_reset.json").read_text()
    )


def patched_sources(source_root: Path) -> dict[str, bytes]:
    """Validate every pinned file before preparing any runtime output."""
    result = {}
    for item in _specification()["files"]:
        source = (source_root / item["path"]).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest == item["after"]:
            result[item["path"]] = source
            continue
        if digest != item["before"]:
            raise ValueError(f"Unsupported FlashInfer source: {item['path']}")
        text = source.decode()
        for old, new in item["replacements"]:
            if text.count(old) != 1:
                raise ValueError(f"Ambiguous FlashInfer patch: {item['path']}")
            text = text.replace(old, new)
        patched = text.encode()
        if hashlib.sha256(patched).hexdigest() != item["after"]:
            raise ValueError(f"Unexpected patched FlashInfer source: {item['path']}")
        result[item["path"]] = patched
    return result


def prepare_runtime(source_root: Path, output: Path) -> dict:
    """Copy a validated runtime atomically into a new, caller-owned directory."""
    source_root = source_root.resolve()
    output = output.absolute()
    if output == source_root or source_root in output.parents:
        raise ValueError("Runtime output must be outside the installed package")
    patches = patched_sources(source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError(f"Refusing to replace existing runtime: {output}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            package = staging / "flashinfer"
            # Dereference package symlinks: overwriting a copied symlink would
            # otherwise mutate the original installation outside this runtime.
            shutil.copytree(
                source_root,
                package,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for relative, content in patches.items():
                (package / relative).write_bytes(content)
            manifest = {
                "upstream": _specification()["upstream"],
                "source_root": str(source_root),
                "files": {
                    relative: hashlib.sha256(content).hexdigest()
                    for relative, content in patches.items()
                },
                "jit_modules": [
                    "sampling_areal_radix_reset_9e032f75",
                    "topk_areal_radix_reset_9e032f75",
                ],
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            os.rename(staging, output)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source
    if source is None:
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or spec.origin is None:
            parser.error("FlashInfer is not installed; pass --source")
        source = Path(spec.origin).parent
    prepare_runtime(source, args.output)


if __name__ == "__main__":
    main()
