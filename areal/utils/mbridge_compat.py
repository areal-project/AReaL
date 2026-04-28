# SPDX-License-Identifier: Apache-2.0
"""mbridge compatibility shims for environments shipping older megatron-core
or missing CUDA-only optional dependencies (e.g. NPU + MindSpeed).

Importing this module installs two shims, both idempotent / no-op when
unnecessary:

1. ``megatron.core.utils.get_tensor_model_parallel_group_if_none`` — added in
   mcore 0.13. mbridge >= 310e8fb imports this at module load time
   (qwen3_vl). On NPU we ship MindSpeed's mcore 0.12.1 which lacks the
   symbol; this shim mirrors the upstream PR
   https://github.com/ISEEKYAN/mbridge/pull/53.

2. ``transformer_engine`` (and ``.pytorch`` / ``.common.recipe``) — required
   by ``megatron.core.extensions.transformer_engine`` (pulled in
   transitively by ``mbridge.models.gemma3``). transformer_engine is
   CUDA-only and not available on Ascend NPU. Register inert stub modules
   so the unconditional ``import transformer_engine as te`` and
   class-statement bases like ``class TELinear(te.pytorch.Linear)`` succeed
   at import time. Anything that actually instantiates these stubs at
   runtime raises a clear error.

Import this module at the top of any AReaL file that does ``import mbridge``
(or any transitive equivalent) so the shims land before mbridge's
``__init__.py`` cascades.
"""

from __future__ import annotations

import sys
import types
import warnings


def _install_get_tensor_model_parallel_group_if_none() -> None:
    import megatron.core.utils as mcore_utils
    from megatron.core import parallel_state

    if hasattr(mcore_utils, "get_tensor_model_parallel_group_if_none"):
        return

    import torch

    def get_tensor_model_parallel_group_if_none(
        tp_group, is_expert: bool = False, check_initialized: bool = True
    ):
        if not torch.distributed.is_initialized():
            return None
        if tp_group is None:
            if torch.distributed.get_rank() == 0:
                warnings.warn(
                    "tp_group is None, using default tp group. "
                    "Passing tp_group will be mandatory soon",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if is_expert:
                tp_group = parallel_state.get_expert_tensor_parallel_group(
                    check_initialized=check_initialized
                )
            else:
                tp_group = parallel_state.get_tensor_model_parallel_group(
                    check_initialized=check_initialized
                )
        return tp_group

    mcore_utils.get_tensor_model_parallel_group_if_none = (
        get_tensor_model_parallel_group_if_none
    )


def _install_transformer_engine_stub() -> None:
    """Register a stub ``transformer_engine`` package if the real one isn't
    importable. Only the surface area touched at module-import time is
    covered: any attribute resolves to a class that supports subclassing
    (``class Foo(te.pytorch.Bar)``) but raises on instantiation.
    """
    if "transformer_engine" in sys.modules:
        return
    try:  # real package available — nothing to do.
        import transformer_engine  # noqa: F401  # type: ignore[import-not-found]

        return
    except ImportError:
        pass

    class _StubMeta(type):
        """Metaclass: any attribute lookup on a stub class returns _StubBase.

        Lets ``te.pytorch.distributed.CudaRNGStatesTracker`` resolve through
        nested attribute chains where ``distributed`` is a class-level stub
        rather than a registered submodule.
        """

        def __getattr__(cls, name: str):
            if name.startswith("_"):
                raise AttributeError(name)
            return _StubBase

    class _StubBase(metaclass=_StubMeta):
        """Inert base class for stubbed TE classes.

        Subclassing is allowed (so ``class Foo(te.pytorch.Linear)`` works at
        import time); instantiation raises a clear error.
        """

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "transformer_engine is not available in this environment "
                "(CUDA-only). The code path that instantiated this class is "
                "unsupported on NPU."
            )

    class _StubModule(types.ModuleType):
        """ModuleType that returns _StubBase for any unknown attribute."""

        def __getattr__(self, name: str):
            if name.startswith("_"):
                raise AttributeError(name)
            return _StubBase

    def _register(path: str) -> _StubModule:
        mod = _StubModule(path)
        sys.modules[path] = mod
        return mod

    te = _register("transformer_engine")
    te.pytorch = _register("transformer_engine.pytorch")
    te.common = _register("transformer_engine.common")
    te.common.recipe = _register("transformer_engine.common.recipe")

    # mcore's get_te_version() reads ``te.__version__`` first, then falls back
    # to ``importlib.metadata.version("transformer-engine")``. The fallback
    # raises PackageNotFoundError on NPU. Report a very high version so all
    # ``is_te_min_version(...)`` checks short-circuit to True; the actual code
    # paths gated on it only run if a TE-using model is instantiated, which
    # would already raise via ``_StubBase.__init__``.
    te.__version__ = "999.0.0"


def apply() -> None:
    _install_transformer_engine_stub()
    _install_get_tensor_model_parallel_group_if_none()


apply()
