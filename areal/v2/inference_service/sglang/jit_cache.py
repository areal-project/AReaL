# SPDX-License-Identifier: Apache-2.0
"""Share SGLang's loaded TVM-FFI CUDA modules between compatible rollout hosts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil


def _runtime_key(cache: Path) -> str | None:
    """Ninja records absolute paths. Reuse a build only in a matching environment."""
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        packages = [version(p) for p in ("torch", "sglang", "apache-tvm-ffi")]
    except (OSError, subprocess.CalledProcessError, PackageNotFoundError):
        return None
    for name in ("sglang-kernel", "sgl-kernel"):
        try:
            packages.append(version(name))
        except PackageNotFoundError:
            pass
    devices = set(gpu.strip().splitlines())
    if len(devices) != 1 or "N/A" in gpu:
        return None
    identity = [
        platform.machine(),
        sys.implementation.cache_tag,
        sys.prefix,
        devices.pop(),
        packages,
        str(cache),
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _loaded_modules(pids: list[int], cache: Path) -> set[Path]:
    modules = set()
    for pid in pids:
        parent = psutil.Process(pid)
        for proc in [parent, *parent.children(recursive=True)]:
            try:
                maps = Path(f"/proc/{proc.pid}/maps").read_text()
            except FileNotFoundError:
                continue
            for line in maps.splitlines():
                if "sgl_kernel_jit_" in line and line.endswith(".so"):
                    module = Path(line.split(maxsplit=5)[-1]).parent
                    if module.parent == cache and module.name.startswith(
                        "sgl_kernel_jit_"
                    ):
                        modules.add(module)
    return modules


def _hashes(module: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in module.iterdir()
        if p.name != "lock" and p.is_file() and not p.is_symlink()
    }


def stage_cache(action: str, shared: Path, pids: list[int]) -> dict:
    # Only accept the controller's private temporary directory under the trial root.
    info = shared.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("Expected a private JIT exchange directory owned by this user")
    cache = (
        Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi"))
        .expanduser()
        .resolve()
    )
    key = _runtime_key(cache)
    if key is None:
        return {"skipped": "CUDA runtime identity unavailable"}
    if action == "export":
        bundle = Path(tempfile.mkdtemp(dir=shared, prefix=key + "-"))
        manifest = {}
        for module in sorted(_loaded_modules(pids, cache)):
            hashes = _hashes(module)
            if ".ninja_log" not in hashes:
                continue
            dest = bundle / module.name
            dest.mkdir()
            for name in hashes:
                shutil.copy2(module / name, dest / name)
            manifest[module.name] = hashes
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return {"exported": len(manifest)}

    installed = []
    for bundle in shared.glob(key + "-*"):
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for source in bundle.glob("sgl_kernel_jit_*"):
            dest = cache / source.name
            if dest.exists():
                continue
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache) as temp:
                candidate = Path(temp) / source.name
                shutil.copytree(source, candidate)
                if _hashes(candidate) != manifest[source.name]:
                    raise ValueError(f"Incomplete JIT module: {source.name}")
                try:
                    candidate.rename(dest)
                except OSError as exc:
                    if exc.errno in (errno.ENOTEMPTY, errno.EEXIST) and dest.is_dir():
                        continue
                    raise
                installed.append(source.name)
    return {"installed": installed}
