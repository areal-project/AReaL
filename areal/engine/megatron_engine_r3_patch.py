# SPDX-License-Identifier: Apache-2.0

"""R3 Integration Patch for MegatronEngine.

Wraps ``MegatronEngine.forward_backward_batch`` to inject per-microbatch
replay setup/teardown around the Megatron pipeline schedule when the
micro-batch data contains ``routed_experts`` tensors.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/megatron")


# ===================================================================
# Public API
# ===================================================================


def patch_megatron_engine_for_r3(
    engine,
    enable_router_replay: bool = False,
) -> None:
    """Patch a ``MegatronEngine`` instance to support Router Replay (R3).

    1. Applies Megatron-Core monkey-patches (TransformerConfig, TopKRouter,
       Dispatcher).
    2. Tags the engine with ``_r3_enabled = True``.
    3. Wraps ``forward_backward_batch`` to inject per-microbatch replay
       setup / teardown around the Megatron pipeline schedule.

    Args:
        engine: A ``MegatronEngine`` instance (already initialized).
        enable_router_replay: Master switch.
    """
    if not enable_router_replay:
        engine._r3_enabled = False
        logger.debug("[R3] Router replay not enabled; skipping engine patch.")
        return

    logger.info("[R3] Patching MegatronEngine for Router Replay.")

    # Mark and save original
    engine._r3_enabled = True
    engine._r3_original_forward_backward_batch = engine.forward_backward_batch
    engine._r3_pending_routed_experts = None

    # Bind the wrapped method
    engine.forward_backward_batch = types.MethodType(_r3_forward_backward_batch, engine)

    logger.debug("[R3] MegatronEngine patched successfully.")


# ===================================================================
# routed_experts alignment (right-padded rollout → left-aligned training)
# ===================================================================


def _align_routed_experts_to_mask(
    routed_experts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    _r3_mb_idx: int | None = None,
) -> torch.Tensor:
    """Align ``routed_experts`` from right-padded rollout format to left-aligned
    training format, matching the token layout implied by ``cu_seqlens``.

    **Rollout format**: ``routed_experts`` is ``(bs, batch_max_seqlen, L, K)``
    with RIGHT padding (real tokens at the BEGINNING of each row, after
    batch-level concatenation; zeros are appended at the end).

    **Training format**: After ``pack_tensor_dict``, tokens are LEFT-aligned
    (real tokens first).  The ``cu_seqlens`` tells us each sample's actual
    length.

    This function extracts the first ``actual_len`` tokens from each
    sample in ``routed_experts`` and produces a ``(bs_aligned, max_seqlen, L, K)``
    tensor with real tokens at the LEFT (matching training convention).

    If ``cu_seqlens`` has more entries than ``routed_experts`` has rows
    (because ``pad_packed_tensor_dict`` appended a dummy padding sequence),
    the output is zero-padded along the batch dimension.

    Args:
        routed_experts: ``(bs, batch_max_seqlen, num_moe_layers, topk)``
        cu_seqlens: ``(n_seqs+1,)`` cumulative sequence lengths.
        max_seqlen: Maximum sequence length (from ``padded_mb["max_seqlen"]``).

    Returns:
        ``(n_seqs, max_seqlen, num_moe_layers, topk)`` aligned tensor.
    """
    re_bs, re_seqlen = routed_experts.shape[:2]
    extra_dims = routed_experts.shape[2:]  # (num_moe_layers, topk)
    n_seqs = cu_seqlens.shape[0] - 1
    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()

    # Output: (n_seqs, max_seqlen, L, K) with real tokens left-aligned
    aligned = torch.zeros(
        n_seqs,
        max_seqlen,
        *extra_dims,
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    )

    for i in range(min(n_seqs, re_bs)):
        actual_len = seq_lens[i]
        if actual_len <= 0:
            continue
        # Source: first actual_len tokens from right-padded routed_experts
        n = min(actual_len, re_seqlen, max_seqlen)
        aligned[i, :n] = routed_experts[i, :n]

    return aligned


# ===================================================================
# routed_experts splitting (robust sample-count inference)
# ===================================================================


def _infer_mb_sample_count(
    mb_dict: dict,
    total_bs: int,
    n_mbs: int,
) -> int:
    """Infer the number of samples in a micro-batch dict.

    Tries multiple strategies in order of reliability:
    1. ``cu_seqlens`` -> ``len(cu_seqlens) - 1`` (packed format, most reliable)
    2. ``attention_mask.shape[0]`` (padded format)
    3. ``input_ids.shape[0]`` (fallback)
    4. Even division (last resort)
    """
    if isinstance(mb_dict, dict):
        # Strategy 1: cu_seqlens (packed format -- most common after pack_tensor_dict)
        cu = mb_dict.get("cu_seqlens")
        if cu is not None:
            return len(cu) - 1

        # Strategy 2: attention_mask (padded format)
        attn = mb_dict.get("attention_mask")
        if attn is not None and hasattr(attn, "shape"):
            return attn.shape[0]

        # Strategy 3: input_ids
        ids = mb_dict.get("input_ids")
        if ids is not None and hasattr(ids, "shape"):
            return ids.shape[0]

    # Strategy 4: last resort
    logger.warning(
        "[R3] _infer_mb_sample_count: no reliable key found, "
        "falling back to even division (%d / %d).",
        total_bs,
        n_mbs,
    )
    return total_bs // n_mbs


def _split_routed_experts_for_mbs(
    routed_experts: torch.Tensor,
    mb_list,
) -> list[torch.Tensor | None]:
    """Split the batch-level ``routed_experts`` tensor into per-micro-batch tensors.

    Uses ``mb_list.forward_indices`` and per-MB sample counts to correctly
    reorder and slice samples, mirroring how ``split_padded_tensor_dict_into_mb_list``
    splits other tensors.

    Args:
        routed_experts: ``(bs, max_seqlen, num_moe_layers, topk)``
        mb_list: ``MicroBatchList`` with ``forward_indices`` and ``group_lens``.

    Returns:
        List of tensors, one per micro-batch, each of shape
        ``(mb_bs, max_seqlen, num_moe_layers, topk)``.
    """
    if routed_experts is None:
        return [None] * len(mb_list)

    forward_indices = mb_list.forward_indices
    n_mbs = len(mb_list)

    if forward_indices is None:
        reordered = routed_experts
    else:
        reordered = routed_experts[forward_indices]

    # Always derive per-micro-batch sample counts from ``mb_list.mbs`` rather
    # than assuming an even ``bs // n_mbs`` split -- the latter silently drops
    # samples when ``bs`` is not divisible by ``n_mbs``.
    result = []
    offset = 0
    for i, mb_dict in enumerate(mb_list.mbs):
        n_samples = _infer_mb_sample_count(mb_dict, routed_experts.shape[0], n_mbs)
        result.append(reordered[offset : offset + n_samples])
        offset += n_samples

    return result


# ===================================================================
# Per-MB cu_seqlens extraction
# ===================================================================


def _get_cu_seqlens_for_mb(mb_item) -> tuple[torch.Tensor, int] | None:
    """Extract ``cu_seqlens`` and ``max_seqlen`` from a ``MicroBatchItem``.

    Prefers ``padded_mb`` (which has the actual TP-aligned dimensions used
    by the model) over ``orig_mb``.

    Returns:
        ``(cu_seqlens, max_seqlen)`` or ``None`` if not available.
    """
    # Try padded_mb first (has TP-aligned cu_seqlens -- this is what the model sees)
    source = None
    if hasattr(mb_item, "padded_mb") and isinstance(mb_item.padded_mb, dict):
        cu = mb_item.padded_mb.get("cu_seqlens")
        max_sl = mb_item.padded_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            source = ("padded_mb", cu, int(max_sl))

    # Try orig_mb as fallback (pre-padding cu_seqlens)
    if (
        source is None
        and hasattr(mb_item, "orig_mb")
        and isinstance(mb_item.orig_mb, dict)
    ):
        cu = mb_item.orig_mb.get("cu_seqlens")
        max_sl = mb_item.orig_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            source = ("orig_mb", cu, int(max_sl))

    if source is None:
        return None

    src_name, cu_out, max_sl_out = source
    return cu_out, max_sl_out


# ===================================================================
# Wrapped forward_backward_batch
# ===================================================================


def _r3_forward_backward_batch(
    self,
    mb_list,
    process_output_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor | None],
    forward_only: bool = False,
) -> None:
    """Drop-in replacement for ``MegatronEngine.forward_backward_batch``
    that injects R3 replay setup around each micro-batch.

    If the data does not contain ``routed_experts``, delegates directly
    to the original method with zero overhead.
    """
    from areal.engine.router_replay_patch import RouterReplay, RouterReplayAction
    from areal.engine.router_replay_utils import (
        RouterReplayHelper,
        clear_router_replay,
        setup_per_microbatch_replay_forward,
    )

    # ------------------------------------------------------------------
    # 1. Retrieve routed_experts.
    # ------------------------------------------------------------------
    routed_experts_batch = None
    _from_side_channel = False

    # Side-channel path (preferred).
    if (
        hasattr(self, "_r3_pending_routed_experts")
        and self._r3_pending_routed_experts is not None
    ):
        routed_experts_batch = self._r3_pending_routed_experts
        self._r3_pending_routed_experts = None  # Consume it
        _from_side_channel = True

    # Strategy B: Legacy path from mb_list.data (backward compatibility)
    if routed_experts_batch is None and not forward_only:
        if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
            routed_experts_batch = mb_list.data.pop("routed_experts", None)

    # Clean from mbs and padded_mbs to avoid confusing downstream code.
    for mb_dict in mb_list.mbs:
        if isinstance(mb_dict, dict):
            mb_dict.pop("routed_experts", None)
    if mb_list.padded_mbs is not None:
        for mb_dict in mb_list.padded_mbs:
            if isinstance(mb_dict, dict):
                mb_dict.pop("routed_experts", None)
    if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
        mb_list.data.pop("routed_experts", None)

    if routed_experts_batch is None:
        return self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )

    logger.info(
        "[R3] forward_backward_batch: %d micro-batches, routed_experts shape=%s, "
        "forward_only=%s",
        len(mb_list),
        routed_experts_batch.shape,
        forward_only,
    )
    # Split routed_experts per micro-batch
    per_mb_routed_experts = _split_routed_experts_for_mbs(routed_experts_batch, mb_list)

    # Store R3 data for the wrapped iterator.
    self._r3_per_mb_experts = per_mb_routed_experts
    self._r3_mb_counter = 0
    model_config = self.tf_config

    # Compute seq_align_to (same as what _prepare_mb_list uses)
    from megatron.core import parallel_state as mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
    seq_align_to = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    # ------------------------------------------------------------------
    # 2b. Set initial replay action to REPLAY_FORWARD.
    # ------------------------------------------------------------------
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    # ------------------------------------------------------------------
    # 3. Wrap the MicroBatchList iterator
    # ------------------------------------------------------------------
    engine_ref = self
    _seq_align_to = seq_align_to

    class _R3MicroBatchIterator:
        """Wraps the micro-batch iterator to inject R3 setup."""

        def __init__(self, base_iter):
            self._base = base_iter

        def __iter__(self):
            return self

        def __next__(self):
            mb_item = next(self._base)

            idx = engine_ref._r3_mb_counter
            engine_ref._r3_mb_counter += 1
            re = (
                engine_ref._r3_per_mb_experts[idx]
                if idx < len(engine_ref._r3_per_mb_experts)
                else None
            )

            # Switch back to REPLAY_FORWARD after backward recompute.
            if RouterReplayHelper.is_replay_backward_action(model_config):
                router_list = RouterReplayHelper.get_micro_batch_router_list(
                    model_config
                )
                for router in router_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            if re is not None:
                # Extract cu_seqlens from padded_mb (TP-aligned, what the model sees)
                cu_info = _get_cu_seqlens_for_mb(mb_item)

                if cu_info is not None:
                    cu_seqlens, max_seqlen = cu_info
                    try:
                        # Use cu_seqlens for alignment instead of
                        # attention_mask. This ensures the packed token order
                        # matches Megatron's actual forward pass.

                        # First, get the ORIGINAL (pre-TP-alignment) cu_seqlens
                        # to know each sample's actual token count for
                        # extracting from routed_experts.
                        orig_cu = None
                        if (
                            hasattr(mb_item, "old_cu_seqlens")
                            and mb_item.old_cu_seqlens is not None
                        ):
                            orig_cu = mb_item.old_cu_seqlens
                        elif hasattr(mb_item, "orig_mb") and isinstance(
                            mb_item.orig_mb, dict
                        ):
                            orig_cu = mb_item.orig_mb.get("cu_seqlens")

                        if orig_cu is None:
                            orig_cu = cu_seqlens

                        # Align from left-padded to left-aligned using original cu_seqlens.
                        aligned_re = _align_routed_experts_to_mask(
                            re,
                            orig_cu,
                            max_seqlen,
                            _r3_mb_idx=idx,
                        )

                        # Pass padded cu_seqlens (TP-aligned) to the model.
                        setup_per_microbatch_replay_forward(
                            aligned_re.to(cu_seqlens.device),
                            cu_seqlens,
                            model_config,
                            seq_align_to=_seq_align_to,
                        )
                    except Exception:
                        logger.warning(
                            "[R3] Failed to setup replay for micro-batch %d.",
                            idx,
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "[R3] Cannot find cu_seqlens for "
                        "micro-batch %d; skipping replay setup. "
                        "Keys in orig_mb: %s, keys in padded_mb: %s.",
                        idx,
                        list(mb_item.orig_mb.keys())
                        if hasattr(mb_item, "orig_mb")
                        and isinstance(mb_item.orig_mb, dict)
                        else "N/A",
                        list(mb_item.padded_mb.keys())
                        if hasattr(mb_item, "padded_mb")
                        and isinstance(mb_item.padded_mb, dict)
                        else "N/A",
                    )
            return mb_item

    # Use a per-instance class swap instead of rebinding the shared
    # ``mb_list.__class__.__iter__``. The latter is a global side effect
    # that also affects any other ``MicroBatchList`` objects alive in the
    # process (e.g. a concurrent engine). Here we create a dynamic
    # subclass whose ``__iter__`` injects the R3 setup logic, and assign
    # it only to *this* instance via ``__class__``. The original class
    # remains untouched. Restoration in the ``finally`` block merely
    # flips ``__class__`` back.
    _r3_original_mb_list_class = mb_list.__class__

    class _R3MicroBatchListProxy(_r3_original_mb_list_class):
        """Per-instance proxy that wraps __iter__ with R3 setup logic."""

        def __iter__(self_inner):
            return _R3MicroBatchIterator(
                _r3_original_mb_list_class.__iter__(self_inner)
            )

    mb_list.__class__ = _R3MicroBatchListProxy

    # Register forward hook for REPLAY_FORWARD → REPLAY_BACKWARD toggle.
    hook_handles = []

    def _r3_post_forward_hook(module, input, output):
        """Switch from REPLAY_FORWARD to REPLAY_BACKWARD after model forward."""
        if RouterReplayHelper.is_replay_forward_action(model_config):
            router_list = RouterReplayHelper.get_micro_batch_router_list(model_config)
            for router in router_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    for model_chunk in self.model:
        handle = model_chunk.register_forward_hook(_r3_post_forward_hook)
        hook_handles.append(handle)

    try:
        self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )
    finally:
        # Remove forward hooks
        for handle in hook_handles:
            handle.remove()
        # Restore the original class on this instance (undo the per-instance
        # class swap done above). The original class was never modified.
        mb_list.__class__ = _r3_original_mb_list_class

        clear_router_replay()
        self._r3_per_mb_experts = None
        self._r3_mb_counter = 0
