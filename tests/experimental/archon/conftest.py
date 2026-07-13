"""Pytest configuration for archon tests.

NOTE: Archon engine requires PyTorch >= 2.9.1.
Qwen3.5-specific tests additionally require transformers >= 5.2.

FP8 tests: stubs heavy optional dependencies (triton, torchao) and uses
lightweight areal package stubs to avoid running areal/__init__.py
(which pulls in heavy infra). This lets pure-logic FP8 unit tests run on
CPU-only environments (macOS, CI without GPU).
"""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Version gates
# ---------------------------------------------------------------------------

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
_MIN_TORCH_VERSION = (2, 9, 1)

try:
    from importlib.metadata import version as _pkg_version

    _TF_VERSION = tuple(int(x) for x in _pkg_version("transformers").split(".")[:2])
except Exception:
    _TF_VERSION = (0, 0)

_MIN_TF_QWEN3_5 = (5, 2)

# Prevent pytest from importing/collecting test files that would fail
collect_ignore_glob: list[str] = []
if _TORCH_VERSION < _MIN_TORCH_VERSION:
    collect_ignore_glob.append("test_*.py")
if _TF_VERSION < _MIN_TF_QWEN3_5:
    collect_ignore_glob.extend(["test_qwen3_5*.py", "test_hf_parity_qwen3_5*.py"])


def pytest_collection_modifyitems(config, items):
    """Skip archon tests based on version requirements."""
    if _TORCH_VERSION >= _MIN_TORCH_VERSION:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Archon tests require PyTorch >= 2.9.1, but found {torch.__version__}"
    )
    for item in items:
        if "experimental/archon" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Lightweight areal package stubs for FP8 unit tests.
#
# Importing areal.api.cli_args transitively pulls in heavy modules
# (areal.engine.fsdp_utils -> transformers.PreTrainedModel, etc.) which may
# not be available on CPU-only environments. We register areal sub-packages
# with real filesystem __path__ (so sub-module .py files can be found) but
# WITHOUT running their __init__.py.
#
# The stubs are applied only when the full areal package hasn't been loaded
# yet (i.e., in lightweight test environments). In full GPU environments
# where areal is properly installed, the stubs are skipped.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent.parent  # /path/to/AReaL


def _pkg_stub(dotted: str, rel_dir: str) -> types.ModuleType:
    """Register a package stub with a real filesystem __path__.

    Using a real __path__ means that `import dotted.submod` will find the
    actual .py file on disk without running the package __init__.py.
    """
    pkg_dir = str(_REPO / rel_dir)
    mod = types.ModuleType(dotted)
    mod.__path__ = [pkg_dir]
    mod.__package__ = dotted
    mod.__file__ = str(_REPO / rel_dir / "__init__.py")
    mod.__spec__ = importlib.machinery.ModuleSpec(
        dotted,
        loader=None,
        origin=mod.__file__,
    )
    mod.__spec__.submodule_search_locations = [pkg_dir]
    sys.modules[dotted] = mod
    return mod


def _bare_stub(name: str) -> types.ModuleType:
    """Register a minimal no-path stub (for leaf modules / blocked packages)."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


# Import transformers before stubbing torchao — transformers probes torchao
# via find_spec at init time; stubbing first corrupts its __init__.
import transformers  # noqa: E402, F401

# Try to import the real areal package. If it succeeds (full GPU environment
# with all deps), skip all areal stubs so that module-level __getattr__
# (lazy imports in areal.api) and other __init__.py logic works correctly.
# If it fails (CPU-only / lightweight environment), apply stubs.
_NEED_AREAL_STUBS = True
try:
    import areal  # noqa: E402, F401

    _NEED_AREAL_STUBS = False
except Exception:
    # Clean up partially-loaded areal modules so stubs can replace them.
    for _key in list(sys.modules):
        if _key == "areal" or _key.startswith("areal."):
            del sys.modules[_key]

if _NEED_AREAL_STUBS:
    # Register areal sub-packages with real __path__ so that
    # `import areal.api.cli_args` resolves to real files without running
    # areal/__init__.py.
    if "areal" not in sys.modules:
        _pkg_stub("areal", "areal")
    if "areal.api" not in sys.modules:
        _pkg_stub("areal.api", "areal/api")
    if "areal.utils" not in sys.modules:
        _pkg_stub("areal.utils", "areal/utils")
    if "areal.experimental" not in sys.modules:
        _pkg_stub("areal.experimental", "areal/experimental")
    if "areal.experimental.models" not in sys.modules:
        _pkg_stub("areal.experimental.models", "areal/experimental/models")
    if "areal.experimental.models.archon" not in sys.modules:
        _pkg_stub(
            "areal.experimental.models.archon", "areal/experimental/models/archon"
        )

    # Stub areal.engine to avoid pulling in heavy deps (fsdp_utils -> transformers).
    if "areal.engine" not in sys.modules:
        _pkg_stub("areal.engine", "areal/engine")
    if "areal.engine.fsdp_utils" not in sys.modules:
        _pkg_stub("areal.engine.fsdp_utils", "areal/engine/fsdp_utils")
    if "areal.engine.megatron_utils" not in sys.modules:
        _pkg_stub("areal.engine.megatron_utils", "areal/engine/megatron_utils")

    # Stub areal.infra to break the circular import chain:
    # cli_args -> areal.utils -> timeutil -> areal.infra.platforms
    # -> areal.infra.__init__ -> rollout_controller -> alloc_mode -> cli_args
    if "areal.infra" not in sys.modules:
        _bare_stub("areal.infra")
    if "areal.infra.platforms" not in sys.modules:
        _bare_stub("areal.infra.platforms")
    sys.modules["areal.infra.platforms"].current_platform = types.SimpleNamespace(
        synchronize=lambda: None,
    )

# ---------------------------------------------------------------------------
# Stub triton and torchao when not available.
# ---------------------------------------------------------------------------


def _try_real_import(name: str) -> bool:
    """Return True if *name* can be imported (not already a stub)."""
    if name in sys.modules:
        return True
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# triton: stub only when not available
if not _try_real_import("triton"):
    for _name in ("triton", "triton.language", "triton.ops"):
        if _name not in sys.modules:
            _bare_stub(_name)

# torchao: stub only when not available
if not _try_real_import("torchao.prototype.blockwise_fp8_training.linear"):
    for _name in (
        "torchao",
        "torchao.quantization",
        "torchao.prototype",
        "torchao.prototype.blockwise_fp8_training",
        "torchao.prototype.blockwise_fp8_training.linear",
    ):
        if _name not in sys.modules:
            _bare_stub(_name)

    # Provide the minimal symbol used by enable_fp8_linear so that the
    # function-level import in fp8.py doesn't crash during collection.
    sys.modules[
        "torchao.prototype.blockwise_fp8_training.linear"
    ].fp8_blockwise_mm = object
