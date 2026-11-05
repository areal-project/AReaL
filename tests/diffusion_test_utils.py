"""Helpers for loading diffusion modules without importing the full package."""

from __future__ import annotations

import importlib.util
import logging as py_logging
import sys
import types
from pathlib import Path

import torch


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if path is not None:
        module.__path__ = [str(path)]
    return module


def load_diffusion_module(module_name: str):
    """Load one diffusion module with only the minimal package scaffolding."""

    repo_root = Path(__file__).resolve().parents[1]
    areal_root = repo_root / "areal"

    _ensure_package("areal", areal_root)
    _ensure_package("areal.experimental", areal_root / "experimental")
    _ensure_package(
        "areal.experimental.diffusion",
        areal_root / "experimental" / "diffusion",
    )

    utils_pkg = _ensure_package("areal.utils")
    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = py_logging.getLogger
    sys.modules["areal.utils.logging"] = logging_mod
    utils_pkg.logging = logging_mod

    if module_name != "diffusion_api":
        load_diffusion_module("diffusion_api")

    full_name = f"areal.experimental.diffusion.{module_name}"
    path = areal_root / "experimental" / "diffusion" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {full_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Compatibility wrapper for tensor closeness across torch versions."""

    if hasattr(torch.testing, "assert_close"):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Tensors are not close.\nactual={actual}\nexpected={expected}"
        )
