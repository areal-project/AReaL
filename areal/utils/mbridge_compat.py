# SPDX-License-Identifier: Apache-2.0
"""mbridge compatibility shims for the NPU + MindSpeed stack.

Importing this module installs two shims, both idempotent / no-op when
unnecessary:

1. ``transformer_engine`` (and ``.pytorch`` / ``.common.recipe``) — required
   by ``megatron.core.extensions.transformer_engine`` (pulled in
   transitively by ``mbridge.models.gemma3``). transformer_engine is
   CUDA-only and not available on Ascend NPU. Register inert stub modules
   so the unconditional ``import transformer_engine as te`` and
   class-statement bases like ``class TELinear(te.pytorch.Linear)`` succeed
   at import time. Anything that actually instantiates these stubs at
   runtime raises a clear error.

2. Re-wrap mbridge's ``@dataclass`` ``TransformerConfig`` subclasses
   (``Qwen3VLTransformerConfig``, ``Qwen2VLTransformerConfig``) with
   MindSpeed's ``transformer_config_init_wrapper``. MindSpeed patches the
   parent ``TransformerConfig.__init__`` to inject CLI args (e.g.
   ``moe_zero_memory_num_layers``) onto the instance before
   ``__post_init__`` reads them. The mbridge subclasses get a fresh
   dataclass-generated ``__init__`` that drops that wrapper, so the
   inherited (still-wrapped) ``__post_init__`` raises ``AttributeError``
   on the missing attribute. Applied via ``apply_post_mbridge()`` because
   the mbridge classes don't exist until ``import mbridge`` runs.

Import this module at the top of any AReaL file that does ``import mbridge``
(or any transitive equivalent) so the shim lands before mbridge's
``__init__.py`` cascades. Then call ``apply_post_mbridge()`` after the
``import mbridge`` line for the second shim.

The mcore<0.13 compatibility shims (``get_tensor_model_parallel_group_if_none``,
``tp_group``/``vp_stage`` kwarg swallowing, ``_preprocess``/``_postprocess``
backports, attention return-tuple padding, NVTX no-ops) lived here when AReaL
ran against MindSpeed's vendored mcore 0.12.1. They were removed when the NPU
stack moved to MindSpeed ``core_r0.16.0`` which vendors mcore 0.16+ natively;
see git history for the previous implementations.
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


def _wrap_mbridge_custom_transformer_configs() -> None:
    """Re-apply MindSpeed's ``transformer_config_init_wrapper`` to mbridge's
    ``@dataclass`` ``TransformerConfig`` subclasses.

    mbridge defines ``Qwen3VLTransformerConfig(TransformerConfig)`` (and
    similar) with its own dataclass fields; the ``@dataclass`` decorator
    regenerates ``__init__``, dropping MindSpeed's wrapper that would
    otherwise inject CLI args (``moe_zero_memory_num_layers`` etc.) onto
    the instance. The inherited ``__post_init__`` is still wrapped and
    calls ``MindSpeedFeaturesManager.pre_validate_features_args(self)``,
    which fails on the missing attribute.

    No-op outside MindSpeed (e.g. CUDA) and idempotent across multiple
    calls.
    """
    try:
        from mindspeed.core.megatron_basic.arguments_basic import (
            transformer_config_init_wrapper,
        )
    except ImportError:
        return  # not on NPU/MindSpeed — nothing to do

    targets: list[type] = []
    try:
        from mbridge.models.qwen3_vl.transformer_config import (
            Qwen3VLTransformerConfig,
        )

        targets.append(Qwen3VLTransformerConfig)
    except ImportError:
        pass
    try:
        from mbridge.models.qwen2_5_vl.transformer_config import (
            Qwen2VLTransformerConfig,
        )

        targets.append(Qwen2VLTransformerConfig)
    except ImportError:
        pass

    for cls in targets:
        if getattr(cls.__init__, "_areal_mindspeed_wrapped", False):
            continue
        wrapped = transformer_config_init_wrapper(cls.__init__)
        wrapped._areal_mindspeed_wrapped = True
        cls.__init__ = wrapped

    # mcore 0.16's ``GPTModel.__init__`` prefers ``self.config.position_embedding_type``
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
            # Re-apply the kwarg the user actually passed (mbridge always
            # passes ``"mrope"`` here).
            self.position_embedding_type = position_embedding_type

        _q3vl_gpt_init._areal_pet_compat = True
        Qwen3VLGPTModel.__init__ = _q3vl_gpt_init


def apply() -> None:
    _install_transformer_engine_stub()


def apply_post_mbridge() -> None:
    """Apply shims that need mbridge to be already imported.

    Call this from any AReaL file that does ``import mbridge`` and may
    trigger ``TransformerConfig`` instantiation downstream.
    """
    _wrap_mbridge_custom_transformer_configs()


apply()
