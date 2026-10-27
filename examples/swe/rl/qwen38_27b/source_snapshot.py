"""Freeze the submitted worktree and private profile without requiring Git on workers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

PROFILE = Path("examples/swe/rl/qwen38_27b/site.env")
NEW_SOURCE_ROOTS = {"areal", "examples", "tests", "scripts"}
CACHE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


def _is_cache(path: Path) -> bool:
    return bool(CACHE_DIRS.intersection(path.parts)) or path.suffix in {".pyc", ".pyo"}


def _source_files(source: Path) -> list[Path]:
    def git_files(*args: str) -> set[Path]:
        output = subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "-z", *args]
        )
        return {Path(os.fsdecode(name)) for name in output.split(b"\0") if name}

    tracked = git_files("--cached")
    untracked = git_files("--others", "--exclude-standard")
    candidates = tracked | {
        path for path in untracked if path.parts[0] in NEW_SOURCE_ROOTS
    }
    return sorted(
        path
        for path in candidates
        if not _is_cache(path)
        and not path.name.endswith(".env")
        and not path.name.startswith(".env.")
        and os.path.lexists(source / path)
    )


def _entry(code: Path, relative: Path) -> dict:
    path = code / relative
    if path.is_symlink():
        target = os.readlink(path)
        if Path(target).is_absolute():
            raise ValueError(f"Snapshot symlink must be relative: {relative}")
        resolved = path.resolve()
        if not resolved.is_relative_to(code):
            raise ValueError(f"Snapshot symlink escapes the code directory: {relative}")
        path.resolve(strict=True)
        return {"symlink": target}
    if not path.is_file():
        raise ValueError(
            f"Snapshot entry must be a file or relative symlink: {relative}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "mode": stat.S_IMODE(path.stat().st_mode)}


def _inventory(code: Path) -> dict:
    return {
        path.relative_to(code).as_posix(): _entry(code, path.relative_to(code))
        for path in sorted(code.rglob("*"))
        if not _is_cache(path.relative_to(code))
        and (path.is_symlink() or not path.is_dir())
    }


def _revision(files: dict) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _copy_path(code: Path, relative: Path) -> Path:
    copied = code / relative
    if any((code / parent).is_symlink() for parent in relative.parents):
        raise ValueError(f"Snapshot destination has a symlink parent: {relative}")
    if copied.is_symlink():
        raise ValueError(f"Snapshot destination is a symlink: {relative}")
    copied.parent.mkdir(parents=True, exist_ok=True)
    return copied


def verify(snapshot: Path, revision: str) -> str:
    """Reject source/profile drift, including extra files, without invoking Git."""
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    code = snapshot / "code"
    if code.is_symlink() or not code.is_dir():
        raise ValueError("Snapshot code must be a directory")
    files = _inventory(code)
    if (
        manifest.get("version") != 1
        or manifest.get("revision") != revision
        or manifest.get("files") != files
        or _revision(files) != revision
    ):
        raise ValueError(
            "Source snapshot manifest mismatch; use a new trial/run directory"
        )
    return revision


def stage(source: Path, profile: Path, destination: Path) -> str:
    """Publish an immutable snapshot or reuse an identical previously staged one."""
    source, profile, destination = (
        path.resolve() for path in (source, profile, destination)
    )
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source checkout and snapshot destination must not overlap")
    git_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    source_files = _source_files(source)
    # Validate the whole selection before copying: tracked descendants may remain
    # in Git's index after a local directory is replaced with a symlink.
    for relative in source_files:
        original = source / relative
        if not original.resolve(strict=True).is_relative_to(source):
            raise ValueError(f"Snapshot symlink escapes the code directory: {relative}")
        if original.is_symlink():
            _entry(source, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        code = temporary / "code"
        code.mkdir()
        for relative in source_files:
            original, copied = source / relative, _copy_path(code, relative)
            if original.is_symlink():
                copied.symlink_to(os.readlink(original))
            else:
                shutil.copyfile(original, copied)
                copied.chmod(stat.S_IMODE(original.stat().st_mode))
        copied_profile = _copy_path(code, PROFILE)
        shutil.copyfile(profile, copied_profile)
        copied_profile.chmod(0o600)
        files = _inventory(code)
        revision = _revision(files)
        manifest = {
            "version": 1,
            "git_revision": git_revision,
            "revision": revision,
            "files": files,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x") as stream:
            manifest_path.chmod(0o600)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Serialize competing submissions; never replace another trial's source.
        with (destination.parent / f".{destination.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists():
                verify(destination, revision)
            else:
                temporary.rename(destination)
        return revision
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    for name in ("source", "profile", "destination"):
        staging.add_argument(f"--{name}", type=Path, required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--snapshot", type=Path, required=True)
    verification.add_argument("--revision", required=True)
    args = parser.parse_args()
    try:
        if args.command == "stage":
            print(stage(args.source, args.profile, args.destination))
        else:
            verify(args.snapshot, args.revision)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Source snapshot error: {exc}\n")


if __name__ == "__main__":
    main()
