# SPDX-License-Identifier: Apache-2.0
"""mbridge compatibility shims for environments shipping older megatron-core
or missing CUDA-only optional dependencies (e.g. NPU + MindSpeed).

Importing this module installs five shims, all idempotent / no-op when
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

3. ``TransformerBlock`` / ``GPTModel`` ``vp_stage`` kwarg — added in
   mcore 0.13. mbridge's Qwen3-VL forwards ``vp_stage`` into both classes
   even when None, so VLM init crashes on mcore 0.12.1. Shim wraps both
   inits to accept and store the kwarg.

4. ``ColumnParallelLinear`` / ``RowParallelLinear`` ``tp_group`` kwarg —
   added in mcore 0.13. mbridge's Qwen3-VL ``Qwen3VLVisionPatchMerger``
   forwards ``tp_group`` to both linear ``build_module`` calls, so VLM
   init crashes on mcore 0.12.1. Shim wraps both inits to discard the
   kwarg.

5. Re-wrap mbridge's ``@dataclass`` ``TransformerConfig`` subclasses
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
(or any transitive equivalent) so the shims land before mbridge's
``__init__.py`` cascades. Then call ``apply_post_mbridge()`` after the
``import mbridge`` line for the third shim.
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


def _install_parallel_linear_tp_group_compat() -> None:
    """Make ``ColumnParallelLinear`` / ``RowParallelLinear`` swallow ``tp_group``.

    mcore 0.13 added a ``tp_group`` kwarg to both linear classes. mbridge's
    ``Qwen3VLVisionPatchMerger`` (qwen3_vl/utils.py) forwards
    ``tp_group=tp_group`` into ``build_module`` for both ``linear_fc1`` and
    ``linear_fc2``, so VLM init crashes on the bundled mcore 0.12.1 with
    ``TypeError: ColumnParallelLinear.__init__() got an unexpected keyword
    argument 'tp_group'``.

    The shim wraps both ``__init__`` methods to discard ``tp_group`` (None
    in our single-TP-group setups). No-op when ``tp_group`` is already
    accepted natively (mcore >= 0.13).
    """
    import inspect

    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    for cls in (ColumnParallelLinear, RowParallelLinear):
        sig = inspect.signature(cls.__init__)
        if "tp_group" in sig.parameters:
            continue  # mcore 0.13+ — natively supported.
        if getattr(cls.__init__, "_areal_tp_group_compat", False):
            continue

        _original_init = cls.__init__

        def _make_init(_original_init):
            def __init__(self, *args, tp_group=None, **kwargs):
                # Silently discard. mbridge resolves None to the default TP
                # group via ``get_tensor_model_parallel_group_if_none`` before
                # passing here; mcore 0.12.1 always uses that default group
                # for collectives, so the value is informational and safe to
                # drop. There is no reliable way to assert "this *is* the
                # default group" without accessing private mpu state.
                _original_init(self, *args, **kwargs)

            __init__._areal_tp_group_compat = True
            return __init__

        cls.__init__ = _make_init(_original_init)


def _install_vp_stage_compat() -> None:
    """Make ``TransformerBlock`` and ``GPTModel`` swallow the ``vp_stage`` kwarg.

    mcore 0.13 added ``vp_stage`` to several class ``__init__`` signatures
    for virtual pipeline parallel support. mcore 0.12.1 (shipped with
    MindSpeed on Ascend NPU) does not. mbridge's Qwen3-VL forwards
    ``vp_stage=vp_stage`` unconditionally into both
    ``TransformerBlock.__init__`` (via ``Qwen3VLVisionTransformerBlock``)
    and ``GPTModel.__init__`` (via ``Qwen3VLGPTModel``), so VLM init
    crashes on NPU.

    This shim wraps each affected ``__init__`` to drop ``vp_stage`` (when
    None — non-None VPP is unsupported here and would be a real bug) and
    stores it on ``self`` so downstream mbridge code that reads
    ``self.vp_stage`` still finds it. No-op when the parameter is already
    accepted natively.
    """
    import inspect

    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_block import TransformerBlock

    for cls in (TransformerBlock, GPTModel):
        sig = inspect.signature(cls.__init__)
        if "vp_stage" in sig.parameters:
            continue  # mcore 0.13+ — natively supported.
        if getattr(cls.__init__, "_areal_vp_stage_compat", False):
            continue

        _original_init = cls.__init__

        def _make_init(_original_init):
            def __init__(self, *args, vp_stage=None, **kwargs):
                if vp_stage is not None:
                    raise NotImplementedError(
                        f"{type(self).__name__}: vp_stage != None requires "
                        "megatron-core >= 0.13; got "
                        f"{vp_stage!r} on the bundled MindSpeed mcore. "
                        "Run with no virtual pipeline parallelism."
                    )
                self.vp_stage = vp_stage
                _original_init(self, *args, **kwargs)

            __init__._areal_vp_stage_compat = True
            return __init__

        cls.__init__ = _make_init(_original_init)


def _install_gpt_model_preprocess_postprocess() -> None:
    """Backport mcore 0.13's ``GPTModel._preprocess`` and ``_postprocess`` to
    the bundled mcore 0.12.1.

    mcore 0.13 split the body of ``GPTModel.forward`` into two helpers so
    subclasses (mbridge's ``Qwen3VLGPTModel``) can wedge custom decoder
    calls between embed/rope (preprocess) and output/loss (postprocess).
    The implementations below mirror mcore 0.12.1's inline ``forward`` body
    so behaviour is unchanged for non-VLM callers.

    No-op when ``_preprocess`` is already defined.
    """
    from megatron.core.models.gpt.gpt_model import GPTModel

    if hasattr(GPTModel, "_preprocess"):
        return

    from collections import OrderedDict

    import torch
    from megatron.core.config_logger import (
        has_config_logger_enabled,
        log_config_to_disk,
    )
    from megatron.core.utils import WrappedTensor, deprecate_inference_params

    # ``BaseInferenceContext`` annotation in mcore 0.13's reference impl is
    # not load-bearing; drop it so the shim doesn't depend on a particular
    # mcore import path.

    def _preprocess(
        self,
        *,
        input_ids,
        position_ids,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
    ):
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
        else:
            decoder_input = None

        # Rotary positional embeddings.
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            if not self.training and self.config.flash_decode and inference_context:
                assert inference_context.is_static_batching(), (
                    "GPTModel currently only supports static inference batching."
                )
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(
                        inference_context.max_sequence_length
                    ),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context,
                    self.decoder,
                    decoder_input,
                    self.config,
                    packed_seq_params,
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == "thd",
                )
        elif (
            self.position_embedding_type == "mrope"
            and not self.config.multi_latent_attention
        ):
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, "
                    "not implemented in MultimodalRotaryEmbedding yet."
                )

        if (
            (self.config.enable_cuda_graph or self.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset]
                * inference_context.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,
            )
        else:
            sequence_len_offset = None

        if (
            inference_context is not None
            and not self.training
            and not has_config_logger_enabled(self.config)
        ):
            decoder_input = WrappedTensor(decoder_input)

        return (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

    def _postprocess(
        self,
        *,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess,
        loss_mask,
        decoder_input,
        attention_mask,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
    ):
        inference_context = deprecate_inference_params(
            inference_context, inference_params
        )

        # Process inference output.
        if inference_context and not inference_context.is_static_batching():
            hidden_states = inference_context.last_token_logits(
                hidden_states.squeeze(1).unsqueeze(0)
            ).unsqueeze(1)

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                compute_language_model_loss=self.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if (
            not self.training
            and inference_context is not None
            and inference_context.is_static_batching()
            and inference_context.materialize_only_last_token_logits
        ):
            hidden_states = hidden_states[-1:, :, :]
        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "decoder_input": decoder_input,
                    "logits": logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix="input_and_logits")

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        return loss

    GPTModel._preprocess = _preprocess
    GPTModel._postprocess = _postprocess


def _nvtx_noop(*_args, **_kwargs) -> None:
    """No-op stub for mcore 0.13's nvtx_range_push/pop helpers.

    NVTX markers are profiling-only; ignoring them on mcore 0.12.1 is safe
    and matches the behaviour of running without nsys.
    """


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

    # mcore 0.13 added ``no_rope_freq`` as a TransformerConfig dataclass
    # field (per-layer mask of layers that skip RoPE). mbridge's Qwen3-VL
    # attention reads ``self.config.no_rope_freq`` at every forward; on
    # mcore 0.12.1 the attribute is absent → AttributeError. Setting None
    # as a class-level fallback makes ``if self.config.no_rope_freq`` skip
    # the no-rope branch — Qwen3-VL doesn't currently use it anyway.
    for cls in targets:
        if not hasattr(cls, "no_rope_freq"):
            cls.no_rope_freq = None

    # mcore 0.12.1's ``GPTModel.__init__`` prefers ``self.config.position_embedding_type``
    # over the constructor kwarg when the attribute exists. MindSpeed's
    # ``transformer_config_init_wrapper`` injects every CLI arg onto every
    # ``TransformerConfig`` instance — including ``position_embedding_type``
    # with default ``'rope'`` — so mbridge's ``Qwen3VLGPTModel(...,
    # position_embedding_type="mrope")`` silently becomes ``"rope"`` and
    # the rope branch in ``_preprocess`` blows up on
    # ``Qwen3VLMultimodalRotaryEmbedding``. Wrap ``Qwen3VLGPTModel.__init__``
    # to overwrite ``self.position_embedding_type`` with ``"mrope"`` after
    # super().__init__.
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

    # mcore 0.13 split ``GPTModel.forward`` into ``_preprocess`` (embed +
    # rope) and ``_postprocess`` (output + loss) so subclasses can wedge in
    # custom decoder calls. mbridge's Qwen3-VL uses both helpers — on
    # mcore 0.12.1 they don't exist (``AttributeError: 'Qwen3VLGPTModel'
    # object has no attribute '_preprocess'``). Inject 0.13-equivalent
    # implementations onto ``GPTModel`` so mbridge's ``Qwen3VLGPTModel.forward``
    # works without changes.
    _install_gpt_model_preprocess_postprocess()

    # mcore 0.13 changed ``Attention._adjust_key_value_for_inference`` to
    # return 6 values (added ``block_table``); 0.12.1 returns 5. mbridge's
    # Qwen3-VL attention unpacks 6 (``query, key, value, rotary_pos_emb,
    # attn_mask_type, block_table``) — ``ValueError: not enough values to
    # unpack (expected 6, got 5)``. Wrap the method to append ``None``.
    try:
        from megatron.core.transformer.attention import Attention as _McoreAttention
    except ImportError:
        _McoreAttention = None
    if _McoreAttention is not None and not getattr(
        _McoreAttention._adjust_key_value_for_inference,
        "_areal_block_table_compat",
        False,
    ):
        _orig_adjust = _McoreAttention._adjust_key_value_for_inference

        def _adjust_key_value_for_inference(self, *args, **kwargs):
            ret = _orig_adjust(self, *args, **kwargs)
            if isinstance(ret, tuple) and len(ret) == 5:
                # Only pad to 6-tuple when the caller is mbridge's Qwen3-VL /
                # Qwen2.5-VL attention (those subclasses unpack 6 values via
                # ``query, ..., attn_mask_type, block_table = self._adjust...``).
                # Base ``Attention.forward`` in MindSpeed-patched mcore 0.12
                # unpacks 5; appending None there breaks unpacking. Use the
                # owning class's module path as a simple discriminator.
                cls_module = getattr(self.__class__, "__module__", "") or ""
                if "qwen" in cls_module and (
                    "qwen3_vl" in cls_module
                    or "qwen2_5_vl" in cls_module
                    or "qwen2_vl" in cls_module
                    or "_vl" in cls_module
                ):
                    return (*ret, None)  # block_table=None
            return ret

        _adjust_key_value_for_inference._areal_block_table_compat = True
        _McoreAttention._adjust_key_value_for_inference = (
            _adjust_key_value_for_inference
        )

    # mcore 0.13 added ``nvtx_range_push`` / ``nvtx_range_pop`` helpers in
    # ``megatron.core.transformer.attention`` (re-exported from
    # ``megatron.core.utils``). mbridge's ``qwen3_vl/attention.py`` does
    # ``from megatron.core.transformer.attention import *`` and calls them
    # in every forward (``nvtx_range_push(suffix="qkv")`` etc.). On
    # mcore 0.12.1 those helpers don't exist anywhere, so the names resolve
    # to ``NameError`` at first forward. Inject inert no-op stubs into the
    # mbridge module namespace so profiling-only calls succeed silently.
    try:
        from mbridge.models.qwen3_vl import attention as _qwen3vl_attn
    except ImportError:
        return
    for fname in ("nvtx_range_push", "nvtx_range_pop"):
        if not hasattr(_qwen3vl_attn, fname):
            setattr(_qwen3vl_attn, fname, _nvtx_noop)


def apply() -> None:
    _install_transformer_engine_stub()
    _install_get_tensor_model_parallel_group_if_none()
    _install_vp_stage_compat()
    _install_parallel_linear_tp_group_compat()


def apply_post_mbridge() -> None:
    """Apply shims that need mbridge to be already imported.

    Call this from any AReaL file that does ``import mbridge`` and may
    trigger ``TransformerConfig`` instantiation downstream.
    """
    _wrap_mbridge_custom_transformer_configs()


apply()
