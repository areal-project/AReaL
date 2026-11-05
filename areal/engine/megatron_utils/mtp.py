# SPDX-License-Identifier: Apache-2.0

"""Independent MTP supervision for logits-based Megatron training."""

from __future__ import annotations

import contextvars
from functools import wraps

import torch

_MTP_BATCH = contextvars.ContextVar("areal_mtp_batch", default=None)

_MTP_TRAIN_LABELS = contextvars.ContextVar("areal_mtp_labels", default=None)
_MTP_TRAIN_LOSS_MASK = contextvars.ContextVar("areal_mtp_mask", default=None)


def _wrap_forward_for_mtp_kwargs(model_cls) -> None:
    """Carry auxiliary targets through logits-only model forwards."""
    _orig_forward = model_cls.forward

    @wraps(_orig_forward)
    def _patched_forward(self, *args, **kwargs):
        mtp_kwargs = kwargs.pop("mtp_kwargs", None)
        mtp_labels = mtp_kwargs.get("mtp_labels") if mtp_kwargs else None
        mtp_loss_mask = mtp_kwargs.get("mtp_loss_mask") if mtp_kwargs else None
        # Multimodal wrappers pass precomputed decoder_input and deliberately
        # set language_model.input_ids=None. MCore still needs token IDs inside
        # the MTP block to build the shifted token embeddings, so restore the
        # layout-aligned IDs from the independent MTP channel in that case.
        if (
            mtp_labels is not None
            and "input_ids" in kwargs
            and kwargs["input_ids"] is None
        ):
            kwargs["input_ids"] = mtp_labels
        batch_token = _MTP_BATCH.set(mtp_kwargs.get("batch") if mtp_kwargs else None)
        labels_token = _MTP_TRAIN_LABELS.set(mtp_labels)
        mask_token = _MTP_TRAIN_LOSS_MASK.set(mtp_loss_mask)
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            _MTP_TRAIN_LOSS_MASK.reset(mask_token)
            _MTP_TRAIN_LABELS.reset(labels_token)
            _MTP_BATCH.reset(batch_token)

    model_cls.forward = _patched_forward


def _roll_mtp_labels(
    mtp_labels,
    roll_tensor,
    *,
    cp_group=None,
    packed_seq_params=None,
):
    """Align raw input IDs with MCore's next-token label convention."""
    roll_kwargs = {
        "shifts": -1,
        "dims": -1,
        "cp_group": cp_group,
        "packed_seq_params": packed_seq_params,
    }
    rolled_labels, _ = roll_tensor(mtp_labels.clone(), **roll_kwargs)
    return rolled_labels


def _detach_mtp_output_weight(output_layer, output_weight):
    """Detach the effective output weight for both tied and untied models."""
    if output_weight is None:
        # MCore only passes output_weight explicitly when the LM head is tied
        # to the input embedding. Untied models instead let output_layer use
        # its internal trainable weight when weight=None.
        output_weight = getattr(output_layer, "weight", None)
    if output_weight is None:
        raise RuntimeError(
            "MTP gradient isolation requires an explicit shared output weight "
            "or an output layer with an internal weight."
        )
    return output_weight.detach()


def _mtp_loss(
    hidden_states,
    labels,
    loss_mask,
    output_layer,
    output_weight,
    runtime_gather_output,
    is_training,
    compute_language_model_loss,
    config,
    cp_group=None,
    packed_seq_params=None,
    scale_logits_fn=None,
):
    """Compute single-layer MTP loss with CP-invariant token normalization."""
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        roll_tensor,
    )

    main_hidden, mtp_hidden = torch.chunk(hidden_states, 2, dim=0)
    raw_labels = _MTP_TRAIN_LABELS.get()
    mask = _MTP_TRAIN_LOSS_MASK.get()
    if mask is None:
        raise ValueError("MTP training requires a loss mask")
    targets = _roll_mtp_labels(
        raw_labels, roll_tensor, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    roll_kwargs = dict(
        shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    targets, _ = roll_tensor(targets, **roll_kwargs)
    mask, _ = roll_tensor(mask.clone(), **roll_kwargs)
    logits, _ = output_layer(
        mtp_hidden,
        weight=_detach_mtp_output_weight(output_layer, output_weight),
        runtime_gather_output=runtime_gather_output,
    )
    if scale_logits_fn is not None:
        logits = scale_logits_fn(logits)
    losses = compute_language_model_loss(targets, logits) * mask
    batch = _MTP_BATCH.get()
    if batch is None:
        raise ValueError("MTP training requires whole-batch normalization")
    if is_training:
        batch["metrics"].add_(
            torch.stack((losses.detach().float().sum(), mask.sum().float()))
        )
    return MTPLossAutoScaler.apply(
        main_hidden, losses * config.mtp_loss_scaling_factor * batch["scale"]
    )


def prepare_mtp_mask(
    mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    old_cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude sequence ends and count valid two-token-ahead targets."""
    mask = mask.clone()
    if cu_seqlens is None:
        mask[..., -1] = 0
        return mask, mask[..., 1:].sum().float()
    starts = cu_seqlens[:-1].long()
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if old_cu_seqlens is not None:
        lengths = old_cu_seqlens[1:] - old_cu_seqlens[:-1]
    mask[cu_seqlens[1:].long() - 1] = 0
    mask[starts[: lengths.numel()] + lengths.long() - 1] = 0
    return mask, (mask.sum() - mask[starts].sum()).float()


def mtp_backward_scale(
    main_tokens: torch.Tensor,
    mtp_tokens: torch.Tensor,
    *,
    cp_size: int,
    dp_cp_size: int,
    num_microbatches: int,
    per_token: bool,
) -> torch.Tensor:
    """Cancel MCore schedule/DDP normalization for a global MTP token mean."""
    numerator = main_tokens / cp_size if per_token else dp_cp_size * num_microbatches
    return numerator / mtp_tokens.clamp(min=1)


class _IsolatedEmbedding(torch.autograd.Function):
    """Keep PP/DDP gradient-ready hooks while blocking auxiliary gradients."""

    @staticmethod
    def forward(ctx, embeddings):
        return embeddings

    @staticmethod
    def backward(ctx, gradient):
        return torch.zeros_like(gradient)


def prepare_mtp_models_for_ddp(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Exclude unused tied-output copies from DDP while retaining Bridge loading."""
    for model in models:
        for module in model.modules():
            if not all(
                getattr(module, flag, False)
                for flag in (
                    "mtp_process",
                    "post_process",
                    "share_embeddings_and_output_weights",
                )
            ):
                continue
            output_weight = getattr(
                getattr(module, "output_layer", None), "weight", None
            )
            if (
                output_weight is not None
                and output_weight is not module.shared_embedding_or_output_weight()
            ):
                # MCore uses the MTP stage's embedding for both projections, but
                # Bridge still loads the separately allocated output weight.
                output_weight.requires_grad_(False)
    return models


def patch_mtp_training() -> None:
    """Install the auxiliary channel without changing ordinary forward calls."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLGPTModel,
    )
    from megatron.core.models.gpt import gpt_model
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )

    if getattr(gpt_model, "_areal_mtp_training_applied", False):
        return
    _wrap_forward_for_mtp_kwargs(gpt_model.GPTModel)
    _wrap_forward_for_mtp_kwargs(Qwen3VLGPTModel)
    original_loss = gpt_model.process_mtp_loss

    @wraps(original_loss)
    def process_loss(*args, **kwargs):
        if _MTP_TRAIN_LABELS.get() is None:
            return original_loss(*args, **kwargs)
        return _mtp_loss(*args, **kwargs)

    original_embeddings = MultiTokenPredictionLayer._get_embeddings

    @wraps(original_embeddings)
    def get_embeddings(self, *args, **kwargs):
        result = original_embeddings(self, *args, **kwargs)
        if _MTP_TRAIN_LABELS.get() is None:
            return result
        # MCore 0.18 also returns padding_mask before the last two tensors.
        return (
            *result[:-2],
            _IsolatedEmbedding.apply(result[-2]),
            result[-1].detach().requires_grad_(True),
        )

    MultiTokenPredictionLayer._get_embeddings = get_embeddings
    gpt_model.process_mtp_loss = process_loss
    gpt_model._areal_mtp_training_applied = True
