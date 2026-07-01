# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for megatron-bridge bugs not yet in a released version.

Each patch is keyed to an upstream PR. Patches are not version-gated; instead
each one's hot path becomes a no-op once the upstream fix is present (the patch
checks for the missing attribute/behavior before acting), and an idempotency
sentinel prevents double-application. Apply patches at import time via
``_apply_patches_on_import()`` at module bottom.
"""

from __future__ import annotations

import contextvars

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")

# Carries the per-forward MTP label tensor from ``GPTModel.forward`` (where the
# caller passes ``mtp_kwargs``) down to ``process_mtp_loss`` (invoked inside
# ``_postprocess``) without touching the long forward signature. A ContextVar is
# PP/recompute-safe: each rank-process has its own value and the forward call is
# synchronous, so set/read/reset stay paired within a single forward.
_MTP_TRAIN_LABELS: contextvars.ContextVar = contextvars.ContextVar(
    "areal_mtp_train_labels", default=None
)


def _patch_qwen3vl_pr3143_word_embeddings() -> None:
    """megatron-bridge PR #3143: expose word_embeddings on MTP shadow embedding.

    Bug (issue #3112 / PR #3143): in ``Qwen3VLGPTModel.forward``, when
    ``mtp_process and sequence_parallel`` are both True, ``self.embedding`` is
    temporarily replaced with a plain closure ``_sp_scatter_embedding``. The
    closure lacks the ``word_embeddings`` attribute that
    ``shared_embedding_or_output_weight()`` accesses during ``_postprocess``
    when ``share_embeddings_and_output_weights=True`` — typical for the
    smaller Qwen3.5 dense models (0.8B/2B/4B).

    Failure mode:
        ``AttributeError: 'function' object has no attribute 'word_embeddings'``

    Affected versions: megatron-bridge 0.4.0 and 0.4.1. Fixed on ``main``
    by commit 20749b09 (PR #3143) but not in any non-alpha release yet.

    Strategy: wrap ``Qwen3VLGPTModel._postprocess`` so it lazily restores
    ``word_embeddings`` on the shadow embedding by inspecting its closure.
    Closure-based recovery is non-invasive — we don't touch ``forward``
    itself (~70 LoC method).
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
            Qwen3VLGPTModel,
        )
    except ImportError:
        return

    if getattr(Qwen3VLGPTModel, "_areal_pr3143_applied", False):
        return

    _orig_postprocess = Qwen3VLGPTModel._postprocess

    def _patched_postprocess(self, *args, **kwargs):
        emb = self.__dict__.get("embedding")
        # Only intervene when the shadow closure is currently installed and
        # lacks the expected attribute.
        if (
            callable(emb)
            and not hasattr(emb, "word_embeddings")
            and emb.__closure__ is not None
        ):
            for cell in emb.__closure__:
                try:
                    target = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(target, "word_embeddings"):
                    emb.word_embeddings = target.word_embeddings
                    break
        return _orig_postprocess(self, *args, **kwargs)

    Qwen3VLGPTModel._postprocess = _patched_postprocess
    Qwen3VLGPTModel._areal_pr3143_applied = True
    logger.info(
        "Applied megatron-bridge PR #3143 workaround: "
        "Qwen3VLGPTModel shadow embedding word_embeddings restoration."
    )


def _patch_gpt_model_mtp_training() -> None:
    """Enable training the Multi-Token-Prediction (MTP) head as an auxiliary loss.

    AReaL computes the *main* loss outside Megatron from logits, so
    ``GPTModel.forward`` is always called with ``labels=None`` and must keep
    returning logits. Megatron-Core 0.17.0's ``process_mtp_loss`` instead derives
    the MTP labels from the main ``labels`` and early-returns when it is None, so
    the MTP loss never fires under AReaL. It also feeds the *shared* output weight
    into the MTP output layer un-detached, which would leak MTP gradients into the
    backbone.

    This patch mirrors slime's ``docker/patch/latest/megatron.patch`` but adapts to
    0.17.0's refactor (MTP loss extracted into the standalone ``process_mtp_loss``):

    1. ``GPTModel.forward`` accepts an extra ``mtp_kwargs={"mtp_labels": ...}`` and
       stashes the labels in a ContextVar; the main ``labels`` stays None so the
       main path keeps emitting logits.
    2. The module-level ``process_mtp_loss`` (looked up at call time inside
       ``_postprocess``) is wrapped so that, when MTP labels are present, it:
         - pre-rolls the labels once (slime parity: 0.17.0 only rolls inside its
           per-layer loop, so MTP layer 0 would otherwise predict t+1 instead of
           t+2); and
         - detaches the shared ``output_weight`` to isolate backbone gradients.
    3. ``MultiTokenPredictionLayer._get_embeddings`` is wrapped to detach the
       backbone hidden states and the MTP embedding input, so the MTP loss only
       updates MTP parameters.
    """
    try:
        from megatron.core.models.gpt import gpt_model
        from megatron.core.transformer import multi_token_prediction as mtp_mod
    except ImportError:
        return

    if getattr(gpt_model, "_areal_mtp_training_applied", False):
        return

    roll_tensor = mtp_mod.roll_tensor
    GPTModel = gpt_model.GPTModel
    MultiTokenPredictionLayer = mtp_mod.MultiTokenPredictionLayer

    # --- 1. GPTModel.forward: capture mtp_kwargs into the ContextVar ---
    _orig_forward = GPTModel.forward

    def _patched_forward(self, *args, **kwargs):
        mtp_kwargs = kwargs.pop("mtp_kwargs", None)
        mtp_labels = mtp_kwargs.get("mtp_labels") if mtp_kwargs else None
        token = _MTP_TRAIN_LABELS.set(mtp_labels)
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            _MTP_TRAIN_LABELS.reset(token)

    GPTModel.forward = _patched_forward

    # --- 2. process_mtp_loss: inject mtp_labels (pre-rolled) + detach weight ---
    _orig_process_mtp_loss = gpt_model.process_mtp_loss

    def _patched_process_mtp_loss(*args, **kwargs):
        mtp_labels = _MTP_TRAIN_LABELS.get()
        if mtp_labels is not None:
            # Roll once up front so the in-loop roll inside the original
            # process_mtp_loss aligns MTP layer 0 to the t+2 target.
            rolled, _ = roll_tensor(
                mtp_labels.clone(),
                shifts=-1,
                dims=-1,
                cp_group=kwargs.get("cp_group"),
                packed_seq_params=kwargs.get("packed_seq_params"),
            )
            kwargs["labels"] = rolled
            output_weight = kwargs.get("output_weight")
            if output_weight is not None:
                kwargs["output_weight"] = output_weight.detach()
        return _orig_process_mtp_loss(*args, **kwargs)

    gpt_model.process_mtp_loss = _patched_process_mtp_loss

    # --- 3. MTP layer embeddings: cut backbone from the MTP graph ---
    _orig_get_embeddings = MultiTokenPredictionLayer._get_embeddings

    def _patched_get_embeddings(self, *args, **kwargs):
        out = _orig_get_embeddings(self, *args, **kwargs)
        if _MTP_TRAIN_LABELS.get() is None:
            return out
        input_ids, position_ids, decoder_input, hidden_states = out
        # detach the shared embedding output and the backbone hidden states so
        # only MTP-internal parameters receive gradients.
        decoder_input = decoder_input.detach()
        hidden_states = hidden_states.detach().requires_grad_(True)
        return input_ids, position_ids, decoder_input, hidden_states

    MultiTokenPredictionLayer._get_embeddings = _patched_get_embeddings

    gpt_model._areal_mtp_training_applied = True
    logger.info(
        "Applied MTP training patch: GPTModel.forward accepts mtp_kwargs, "
        "process_mtp_loss uses an independent label channel with detached output "
        "weight, and MTP gradients are isolated from the backbone."
    )


def _apply_patches_on_import() -> None:
    _patch_qwen3vl_pr3143_word_embeddings()
    _patch_gpt_model_mtp_training()


_apply_patches_on_import()
