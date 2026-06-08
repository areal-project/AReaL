"""Pre-import hook for Python < 3.12 compatibility.

The top-level ``areal/__init__.py`` imports from modules that use PEP 695 syntax
(``def func[T](...)``) which is only valid on Python 3.12+. When running tests on
Python 3.10/3.11 we register lightweight stubs for the ``areal`` namespace packages
so that importing ``areal.experimental.weight_update.*`` never triggers the
problematic top-level init.
"""

from __future__ import annotations

import os
import sys
import types

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_STUB_PACKAGES = [
    ("areal", os.path.join(_REPO_ROOT, "areal")),
    ("areal.experimental", os.path.join(_REPO_ROOT, "areal", "experimental")),
    (
        "areal.experimental.weight_update",
        os.path.join(_REPO_ROOT, "areal", "experimental", "weight_update"),
    ),
]


def _ensure_namespace_stubs():
    """Insert namespace-style modules for ``areal`` ancestors with real paths."""
    for name, path in _STUB_PACKAGES:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    for name, _path in _STUB_PACKAGES:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_attr = parts
            parent = sys.modules.get(parent_name)
            child = sys.modules.get(name)
            if parent is not None and child is not None:
                setattr(parent, child_attr, child)


if sys.version_info < (3, 12):
    _ensure_namespace_stubs()
