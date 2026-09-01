# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims needed before and after importing mbridge.

Importing this module installs two shims, both idempotent / no-op when
unnecessary:

1. Register inert ``transformer_engine`` modules when TE is unavailable so
   mbridge's unconditional import-time class definitions remain importable.
   Real CUDA TE and TransformerEngineNPU make this a no-op.

2. Preserve Qwen3-VL's requested ``position_embedding_type`` after MindSpeed
   injects argument defaults into the MCore config.

Import this module at the top of any AReaL file that does ``import mbridge``
(or any transitive equivalent) so the shim lands before mbridge's
``__init__.py`` cascades. Then call ``apply_post_mbridge()`` after the
``import mbridge`` line for the second shim.

Older MCore API shims and MindSpeed transformer-config rewrapping are no
longer needed by the pinned MCore 0.18 / MegatronAdaptor stack.
"""

from __future__ import annotations

import sys
import types


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


def _patch_qwen3vl_position_embedding_type() -> None:
    """Preserve mbridge's requested Qwen3-VL position embedding type."""
    # MCore prefers ``self.config.position_embedding_type``
    # over the constructor kwarg when the attribute exists (gpt_model.py:128-131).
    # MindSpeed's ``transformer_config_init_wrapper`` injects every CLI arg onto
    # every ``TransformerConfig`` instance — including ``position_embedding_type``
    # with default ``'rope'`` — so mbridge's ``Qwen3VLGPTModel(...,
    # position_embedding_type="mrope")`` silently becomes ``"rope"`` and the
    # ``elif self.position_embedding_type == 'mrope'`` branch is skipped, leaving
    # ``self.rotary_pos_emb`` unbound for the Qwen3-VL multimodal RoPE.
    # Wrap ``Qwen3VLGPTModel.__init__`` to overwrite ``self.position_embedding_type``
    # with the caller's kwarg after super().__init__.
    try:
        from mbridge.models.qwen3_vl.gpt_model import Qwen3VLGPTModel
    except ImportError:
        Qwen3VLGPTModel = None
    if Qwen3VLGPTModel is not None and not getattr(
        Qwen3VLGPTModel.__init__, "_areal_pet_compat", False
    ):
        _orig_q3_gpt_init = Qwen3VLGPTModel.__init__

        def _q3vl_gpt_init(self, *args, position_embedding_type="rope", **kwargs):
            _orig_q3_gpt_init(
                self, *args, position_embedding_type=position_embedding_type, **kwargs
            )
            self.position_embedding_type = position_embedding_type

        _q3vl_gpt_init._areal_pet_compat = True
        Qwen3VLGPTModel.__init__ = _q3vl_gpt_init


def apply() -> None:
    _install_transformer_engine_stub()


def apply_post_mbridge() -> None:
    """Apply shims that need mbridge classes to be defined."""
    _patch_qwen3vl_position_embedding_type()


apply()
