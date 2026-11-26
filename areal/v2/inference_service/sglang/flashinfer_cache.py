# SPDX-License-Identifier: Apache-2.0
"""Move compatible FlashInfer and CUDA driver caches within one Ray job."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .jit_cache import _runtime_key

_AOT_MODULES = frozenset({"fmha_gen", "sampling"})


def supports_cache_staging() -> bool:
    """Limit the extra warmup to the supported FlashInfer runtime."""
    try:
        return (
            version("sglang") == "0.5.20" and version("flashinfer-python") == "0.6.18"
        )
    except PackageNotFoundError:
        return False


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(path: Path):
    if path.is_symlink():
        return
    for directory, dirs, names in os.walk(path):
        base = Path(directory)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in names:
            entry = base / name
            if (
                not name.endswith(".lock")
                and entry.is_file()
                and not entry.is_symlink()
            ):
                yield entry


def _module_dirs(root: Path, release: str):
    for variant in sorted((root / release).glob("*")):
        if not re.fullmatch(r"[0-9a-z_]+", variant.name) or variant.is_symlink():
            continue
        cached_ops = variant / "cached_ops"
        if cached_ops.is_symlink():
            continue
        for module in sorted(cached_ops.glob("*")):
            if module.is_symlink() or not (module / ".ninja_log").is_file():
                continue
            if (module / f"{module.name}.so").is_file():
                yield module
                yield variant / "generated" / module.name


def _export(root: Path, release: str, shared: Path, key: str) -> dict:
    paths = {}
    for directory in (*_module_dirs(root, release), root / "cubins"):
        paths.update((item.relative_to(root), item) for item in _files(directory))
    with tempfile.TemporaryDirectory(prefix=".flashinfer-", dir=shared) as stage:
        package = Path(stage) / "bundle"
        package.mkdir()
        manifest = {}
        for relative, source in paths.items():
            dest = package / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            manifest[relative.as_posix()] = _digest(dest)
        (package / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        bundle_id = Path(stage).name.removeprefix(".flashinfer-")
        package.rename(shared / f"flashinfer-{key}-{bundle_id}")
    return {
        "exported": len(manifest),
        "modules": sorted(
            relative.stem for relative in paths if relative.suffix == ".so"
        ),
    }


def _publish_aot(root: Path, release: str, target: Path) -> str | None:
    parts = target.relative_to(root).parts
    if release != "0.6.18" or len(parts) != 5 or parts[2] != "cached_ops":
        return None
    _, arch, _, module, filename = parts
    if module not in _AOT_MODULES or filename != f"{module}.so":
        return None
    prebuilt = root / release / arch / "areal_aot" / module / filename
    prebuilt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=prebuilt.parent) as stage:
        candidate = Path(stage) / filename
        shutil.copy2(target, candidate)
        os.replace(candidate, prebuilt)
    return module


def _install(root: Path, release: str, shared: Path, key: str) -> dict:
    installed = 0
    reused = 0
    modules = set()
    aot_modules = set()
    for bundle in sorted(shared.glob(f"flashinfer-{key}-*")):
        if bundle.is_symlink():
            continue
        manifest = json.loads((bundle / "manifest.json").read_text())
        sources = []
        for relative, expected in manifest.items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] not in (release, "cubins")
            ):
                raise ValueError(f"Invalid FlashInfer cache member: {relative}")
            source = bundle / path
            target = root / path
            if not source.resolve().is_relative_to(
                bundle.resolve()
            ) or not target.resolve().is_relative_to(root):
                raise ValueError(
                    f"FlashInfer cache member resolves outside its cache: {relative}"
                )
            if _digest(source) != expected:
                raise ValueError(f"Corrupt FlashInfer cache member: {relative}")
            sources.append((source, target, expected))
        for source, target, expected in sources:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _digest(target) == expected:
                    reused += 1
                    if target.suffix == ".so":
                        modules.add(target.stem)
                    if (name := _publish_aot(root, release, target)) is not None:
                        aot_modules.add(name)
                continue
            with tempfile.TemporaryDirectory(
                prefix=".install-", dir=target.parent
            ) as temp:
                candidate = Path(temp) / target.name
                shutil.copy2(source, candidate)
                try:
                    os.link(candidate, target)
                except FileExistsError:
                    continue
                installed += 1
                if target.suffix == ".so":
                    modules.add(target.stem)
                if (name := _publish_aot(root, release, target)) is not None:
                    aot_modules.add(name)
    return {
        "installed": installed,
        "reused": reused,
        "modules": sorted(modules),
        "aot_modules": sorted(aot_modules),
    }


def enable_staged_aot() -> None:
    """Use verified job-local libraries through FlashInfer's AOT load path."""
    if not supports_cache_staging():
        return
    from flashinfer.jit import env
    from flashinfer.jit.core import JitSpecNvcc

    directory = env.FLASHINFER_WORKSPACE_DIR / "areal_aot"
    if not directory.is_dir():
        return
    default_path = JitSpecNvcc.aot_path.fget

    def aot_path(spec):
        default = default_path(spec)
        if spec.name in _AOT_MODULES and not default.is_file():
            staged = directory / spec.name / f"{spec.name}.so"
            if staged.is_file():
                return staged
        return default

    JitSpecNvcc.aot_path = property(aot_path)


def stage_flashinfer_cache(action: str, shared: Path) -> dict:
    try:
        release = version("flashinfer-python")
    except PackageNotFoundError:
        return {"skipped": "FlashInfer package unavailable"}
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not base:
        return {"skipped": "FlashInfer workspace must be explicit on each Guard"}
    root = (Path(base) / ".cache/flashinfer").resolve()
    runtime = _runtime_key(root)
    if not runtime:
        return {"skipped": "CUDA runtime identity unavailable"}
    identity = [
        runtime,
        release,
        os.environ.get("CUDA_HOME"),
        os.environ.get("FLASHINFER_NVCC"),
    ]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    if action == "export":
        return _export(root, release, shared, key)
    if action == "install":
        return _install(root, release, shared, key)
    raise ValueError(f"Unknown FlashInfer cache action: {action}")


def _driver_entry(relative: str) -> bool:
    return relative == "index" or bool(
        re.fullmatch(r"[a-f0-9]/[a-f0-9]/[a-f0-9]+", relative)
    )


def stage_driver_cache(action: str, shared: Path) -> dict:
    """Copy the driver's opaque index and objects only as a complete snapshot."""
    configured = os.environ.get("CUDA_CACHE_PATH")
    if not configured:
        return {"skipped": "CUDA driver cache directory must be explicit"}
    root = Path(configured).resolve()
    key = _runtime_key(root)
    if not key:
        return {"skipped": "CUDA runtime identity unavailable"}
    if action == "export":
        if not (root / "index").is_file():
            return {"skipped": "Source CUDA driver cache has no index"}
        with tempfile.TemporaryDirectory(prefix=".cuda-driver-", dir=shared) as stage:
            package = Path(stage) / "bundle"
            package.mkdir()
            manifest = {}
            for source in _files(root):
                relative = source.relative_to(root)
                if not _driver_entry(relative.as_posix()):
                    continue
                dest = package / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                manifest[relative.as_posix()] = {
                    "sha256": _digest(dest),
                    "bytes": dest.stat().st_size,
                }
            if "index" not in manifest:
                return {"skipped": "Source CUDA driver cache index is unavailable"}
            (package / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            bundle_id = Path(stage).name.removeprefix(".cuda-driver-")
            package.rename(shared / f"cuda-driver-{key}-{bundle_id}")
        return {
            "exported": len(manifest),
            "bytes": sum(row["bytes"] for row in manifest.values()),
        }
    if action != "install":
        raise ValueError(f"Unknown CUDA driver cache action: {action}")
    if root.exists() and any(root.iterdir()):
        return {"skipped": "Target CUDA driver cache is already in use"}
    bundles = [
        path
        for path in sorted(shared.glob(f"cuda-driver-{key}-*"))
        if not path.is_symlink()
    ]
    if not bundles:
        return {"skipped": "No compatible CUDA driver cache snapshot"}
    bundle = bundles[0]
    manifest = json.loads((bundle / "manifest.json").read_text())
    if "index" not in manifest or not all(
        _driver_entry(relative) for relative in manifest
    ):
        raise ValueError("Invalid CUDA driver cache snapshot")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cuda-driver-install-", dir=root.parent
    ) as stage:
        cache = Path(stage) / "cache"
        cache.mkdir(mode=0o700)
        for relative, detail in manifest.items():
            source = bundle / relative
            if not source.resolve().is_relative_to(bundle.resolve()):
                raise ValueError(
                    f"CUDA driver cache member resolves outside its cache: {relative}"
                )
            dest = cache / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            if _digest(dest) != detail["sha256"]:
                raise ValueError(f"Corrupt CUDA driver cache member: {relative}")
        # rename refuses to overwrite a nonempty cache populated by another process.
        os.replace(cache, root)
    return {
        "installed": len(manifest),
        "bytes": sum(row["bytes"] for row in manifest.values()),
    }
