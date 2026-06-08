#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check consistency between pyproject.toml and pyproject.vllm.toml.

Only packages marked as "escapable" (inference-backend-specific) are allowed
to have different versions or be present in only one file.  Everything else
must be identical across the two project configurations.

Usage::

    python areal/tools/check_pyproject_consistency.py
    python areal/tools/check_pyproject_consistency.py pyproject.toml pyproject.vllm.toml

Exit codes:
    0  — files are consistent
    1  — inconsistencies found (details printed to stderr)
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

# ── Escapable packages ──────────────────────────────────────────────────────
# These packages are expected to differ between the SGLang (default) and vLLM
# variants because the two inference backends pin mutually-incompatible
# versions of torch / torchao / etc.
ESCAPABLE_PACKAGES: frozenset[str] = frozenset(
    {
        "torch",
        "torchao",
        "torchaudio",
        "torchvision",
        "sglang",
        "vllm",
        "nvidia-cudnn-cu12",
        "openai",
        "soundfile",
        "peft",
        "transformers",
    }
)

# Optional-dependency extras that are backend-specific.
# These extras are expected to exist exclusively in one variant.
BACKEND_EXTRAS: frozenset[str] = frozenset({"sglang", "vllm"})

# ── Helpers ─────────────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """PEP 503-normalize a package name (lowercase, ``[-_.]+`` → ``-``)."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _parse_dep_name(dep: str) -> str:
    """Extract the normalized package name from a dependency string.

    Examples::

        "torch==2.9.1+cu129; ..." → "torch"
        "sglang[tracing]==0.5.9"  → "sglang"
        "areal[cuda-train]"       → "areal"
    """
    match = re.match(r"([A-Za-z0-9][-A-Za-z0-9_.]*)", dep.strip())
    if not match:
        return dep.strip()
    return _normalize_name(match.group(1))


def _dep_list_to_dict(deps: list[str]) -> dict[str, list[str]]:
    """Map normalized package name → list of full dependency strings.

    A package may have multiple entries (e.g. platform-specific markers).
    """
    result: dict[str, list[str]] = {}
    for dep in deps:
        name = _parse_dep_name(dep)
        result.setdefault(name, []).append(dep.strip())
    return result


def _is_escapable(name: str) -> bool:
    return _normalize_name(name) in {_normalize_name(n) for n in ESCAPABLE_PACKAGES}


def _is_backend_selfref(dep: str) -> bool:
    """True if *dep* is a self-reference to a backend-specific extra.

    E.g. ``areal[sglang]`` or ``areal[vllm]``.
    """
    match = re.match(r"\s*areal\[([^\]]+)\]", dep.strip())
    return match is not None and match.group(1) in BACKEND_EXTRAS


# ── Comparison engine ───────────────────────────────────────────────────────


class _Checker:
    """Accumulates errors while comparing two parsed TOML dicts."""

    def __init__(self, file_a: str, file_b: str) -> None:
        self.file_a = file_a
        self.file_b = file_b
        self.errors: list[str] = []

    def _err(self, msg: str) -> None:
        self.errors.append(msg)

    # ── generic deep comparison ─────────────────────────────────────────

    def _cmp_values(self, path: str, a: Any, b: Any) -> None:
        if type(a) is not type(b):
            self._err(
                f"{path}: type mismatch — {self.file_a} has "
                f"{type(a).__name__}, {self.file_b} has {type(b).__name__}"
            )
            return
        if isinstance(a, dict):
            self._cmp_dicts(path, a, b)
        elif isinstance(a, list):
            if a != b:
                self._err(
                    f"{path}: lists differ\n  {self.file_a}: {a}\n  {self.file_b}: {b}"
                )
        elif a != b:
            self._err(f"{path}: {self.file_a}={a!r} ≠ {self.file_b}={b!r}")

    def _cmp_dicts(self, path: str, a: dict, b: dict) -> None:
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                self._err(f"{sub}: missing in {self.file_a}")
            elif key not in b:
                self._err(f"{sub}: missing in {self.file_b}")
            else:
                self._cmp_values(sub, a[key], b[key])

    # ── [project].dependencies ──────────────────────────────────────────

    def check_dependencies(self, deps_a: list[str], deps_b: list[str]) -> None:
        dict_a = _dep_list_to_dict(deps_a)
        dict_b = _dep_list_to_dict(deps_b)
        for name in sorted(set(dict_a) | set(dict_b)):
            if _is_escapable(name):
                continue
            in_a, in_b = name in dict_a, name in dict_b
            if in_a and not in_b:
                self._err(
                    f"dependencies: non-escapable package {name!r} "
                    f"in {self.file_a} but missing in {self.file_b}"
                )
            elif in_b and not in_a:
                self._err(
                    f"dependencies: non-escapable package {name!r} "
                    f"in {self.file_b} but missing in {self.file_a}"
                )
            elif sorted(dict_a[name]) != sorted(dict_b[name]):
                self._err(
                    f"dependencies: non-escapable package {name!r} differs\n"
                    f"  {self.file_a}: {dict_a[name]}\n"
                    f"  {self.file_b}: {dict_b[name]}"
                )

    # ── [project.optional-dependencies] ─────────────────────────────────

    def check_optional_deps(
        self,
        extras_a: dict[str, list[str]],
        extras_b: dict[str, list[str]],
    ) -> None:
        all_extras = sorted(set(extras_a) | set(extras_b))
        for extra in all_extras:
            # Backend-specific extras are expected to be exclusive.
            if extra in BACKEND_EXTRAS:
                continue

            if extra not in extras_a:
                self._err(
                    f"optional-dependencies: extra {extra!r} missing in {self.file_a}"
                )
                continue
            if extra not in extras_b:
                self._err(
                    f"optional-dependencies: extra {extra!r} missing in {self.file_b}"
                )
                continue

            # Filter out backend-specific self-references before comparing.
            filtered_a = sorted(
                d for d in extras_a[extra] if not _is_backend_selfref(d)
            )
            filtered_b = sorted(
                d for d in extras_b[extra] if not _is_backend_selfref(d)
            )
            if filtered_a != filtered_b:
                self._err(
                    f"optional-dependencies.{extra}: differs "
                    f"(after filtering backend self-refs)\n"
                    f"  {self.file_a}: {filtered_a}\n"
                    f"  {self.file_b}: {filtered_b}"
                )

    # ── [tool.uv].override-dependencies ─────────────────────────────────

    def check_override_deps(
        self, overrides_a: list[str], overrides_b: list[str]
    ) -> None:
        dict_a = _dep_list_to_dict(overrides_a)
        dict_b = _dep_list_to_dict(overrides_b)
        for name in sorted(set(dict_a) | set(dict_b)):
            if _is_escapable(name):
                continue
            in_a, in_b = name in dict_a, name in dict_b
            if in_a and not in_b:
                self._err(
                    f"tool.uv.override-dependencies: {name!r} "
                    f"in {self.file_a} but missing in {self.file_b}"
                )
            elif in_b and not in_a:
                self._err(
                    f"tool.uv.override-dependencies: {name!r} "
                    f"in {self.file_b} but missing in {self.file_a}"
                )
            elif sorted(dict_a[name]) != sorted(dict_b[name]):
                self._err(
                    f"tool.uv.override-dependencies: "
                    f"non-escapable {name!r} differs\n"
                    f"  {self.file_a}: {dict_a[name]}\n"
                    f"  {self.file_b}: {dict_b[name]}"
                )

    # ── [tool.uv.sources] ──────────────────────────────────────────────

    def check_uv_sources(
        self, sources_a: dict[str, Any], sources_b: dict[str, Any]
    ) -> None:
        for pkg in sorted(set(sources_a) | set(sources_b)):
            if pkg not in sources_a:
                self._err(f"tool.uv.sources: {pkg!r} missing in {self.file_a}")
            elif pkg not in sources_b:
                self._err(f"tool.uv.sources: {pkg!r} missing in {self.file_b}")
            elif _is_escapable(pkg):
                # Just verify index names match (extra names may differ).
                idx_a = sorted(
                    {e.get("index", "") for e in sources_a[pkg] if isinstance(e, dict)}
                )
                idx_b = sorted(
                    {e.get("index", "") for e in sources_b[pkg] if isinstance(e, dict)}
                )
                if idx_a != idx_b:
                    self._err(
                        f"tool.uv.sources.{pkg}: index names differ — "
                        f"{self.file_a}={idx_a}, {self.file_b}={idx_b}"
                    )
            elif sources_a[pkg] != sources_b[pkg]:
                self._err(
                    f"tool.uv.sources.{pkg}: differs (non-escapable)\n"
                    f"  {self.file_a}: {sources_a[pkg]}\n"
                    f"  {self.file_b}: {sources_b[pkg]}"
                )

    # ── [tool.uv] (top-level) ──────────────────────────────────────────

    def check_tool_uv(self, uv_a: dict, uv_b: dict) -> None:
        special_keys = {
            "override-dependencies",
            "sources",
            "conflicts",
            "dependency-metadata",
        }

        self.check_override_deps(
            uv_a.get("override-dependencies", []),
            uv_b.get("override-dependencies", []),
        )
        self.check_uv_sources(
            uv_a.get("sources", {}),
            uv_b.get("sources", {}),
        )
        # `conflicts` may be structurally different (single-backend needs
        # none); skip comparison.

        for key in sorted(set(uv_a) | set(uv_b)):
            if key in special_keys:
                continue
            sub = f"tool.uv.{key}"
            if key not in uv_a:
                self._err(f"{sub}: missing in {self.file_a}")
            elif key not in uv_b:
                self._err(f"{sub}: missing in {self.file_b}")
            else:
                self._cmp_values(sub, uv_a[key], uv_b[key])

    # ── entry point ─────────────────────────────────────────────────────

    def run(self, toml_a: dict, toml_b: dict) -> int:
        # 1. build-system
        self._cmp_values(
            "build-system",
            toml_a.get("build-system", {}),
            toml_b.get("build-system", {}),
        )

        # 2. project metadata (everything except deps)
        proj_a = toml_a.get("project", {})
        proj_b = toml_b.get("project", {})
        skip = {"dependencies", "optional-dependencies"}
        meta_a = {k: v for k, v in proj_a.items() if k not in skip}
        meta_b = {k: v for k, v in proj_b.items() if k not in skip}
        if meta_a != meta_b:
            self._cmp_dicts("project", meta_a, meta_b)

        # 3. dependencies
        self.check_dependencies(
            proj_a.get("dependencies", []),
            proj_b.get("dependencies", []),
        )

        # 4. optional-dependencies
        self.check_optional_deps(
            proj_a.get("optional-dependencies", {}),
            proj_b.get("optional-dependencies", {}),
        )

        # 5. dependency-groups
        self._cmp_values(
            "dependency-groups",
            toml_a.get("dependency-groups", {}),
            toml_b.get("dependency-groups", {}),
        )

        # 6. tool.uv
        self.check_tool_uv(
            toml_a.get("tool", {}).get("uv", {}),
            toml_b.get("tool", {}).get("uv", {}),
        )

        # 7. Other tool sections (pytest, ruff, etc.)
        tool_a = toml_a.get("tool", {})
        tool_b = toml_b.get("tool", {})
        for key in sorted(set(tool_a) | set(tool_b)):
            if key == "uv":
                continue
            sub = f"tool.{key}"
            if key not in tool_a:
                self._err(f"{sub}: missing in {self.file_a}")
            elif key not in tool_b:
                self._err(f"{sub}: missing in {self.file_b}")
            else:
                self._cmp_values(sub, tool_a[key], tool_b[key])

        if self.errors:
            print(
                f"❌ {len(self.errors)} inconsistenc"
                f"{'y' if len(self.errors) == 1 else 'ies'} "
                f"between {self.file_a} and {self.file_b}:",
                file=sys.stderr,
            )
            for e in self.errors:
                print(f"  {e}", file=sys.stderr)
            return 1

        print(
            f"✅ {self.file_a} and {self.file_b} are consistent "
            f"(escapable: {', '.join(sorted(ESCAPABLE_PACKAGES))})"
        )
        return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["pyproject.toml", "pyproject.vllm.toml"],
        help=(
            "Pair of pyproject files to compare "
            "(default: pyproject.toml pyproject.vllm.toml)"
        ),
    )
    args = parser.parse_args(argv)

    if len(args.files) != 2:
        print(
            f"Error: exactly two pyproject files required, got {len(args.files)}",
            file=sys.stderr,
        )
        return 1

    file_a, file_b = args.files
    path_a, path_b = Path(file_a), Path(file_b)

    for p in (path_a, path_b):
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1

    with open(path_a, "rb") as f:
        toml_a = tomllib.load(f)
    with open(path_b, "rb") as f:
        toml_b = tomllib.load(f)

    checker = _Checker(file_a, file_b)
    return checker.run(toml_a, toml_b)


if __name__ == "__main__":
    raise SystemExit(main())
