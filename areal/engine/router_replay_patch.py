# SPDX-License-Identifier: Apache-2.0

"""Monkey-patches for Megatron-Core MoE components to support Router Replay (R3).

Forces TopKRouter to use pre-recorded expert assignments from rollout inference
instead of computing new ones during training, eliminating train/inference routing
mismatch caused by weight staleness in async RL training.
"""

from __future__ import annotations

import inspect
import types
import warnings
from enum import Enum
from functools import wraps

import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/patch")

# ---------------------------------------------------------------------------
# Optional megatron-core imports with fallback
# ---------------------------------------------------------------------------
try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
    )
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
except ImportError:
    apply_router_token_dropping = None
    compute_routing_scores_for_aux_loss = None
    TopKRouter = None
    MoEAlltoAllTokenDispatcher = None
    TransformerConfig = None
    warnings.warn(
        "[R3] Could not import megatron.core MoE components; "
        "some MoE features may be unavailable.",
        stacklevel=2,
    )

try:
    from megatron.core.transformer.moe.moe_utils import group_limited_topk
except ImportError:
    group_limited_topk = None


# ===================================================================
# RouterReplayAction enum and RouterReplay class
# ===================================================================


class RouterReplayAction(Enum):
    """Actions controlling the MoE routing replay behaviour."""

    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """Manages recording and replaying of MoE routing decisions.

    Each MoE layer gets one ``RouterReplay`` instance.  The class-level
    list ``router_instances`` holds all of them so that global operations
    (set action, distribute data, clear state) are straightforward.
    """

    # Class-level list of all router instances (one per MoE layer).
    router_instances: list[RouterReplay] = []

    # Class-level pipeline parallelism size for backward remapping.
    # Set by the engine patch before forward_backward_func.
    pp_size: int = 1

    # ------------------------------------------------------------------
    # Class-level (static) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list) -> None:
        """Distribute per-layer topk indices to ``RouterReplay`` instances.

        Args:
            all_layers_topk_indices: List of tensors, one per MoE layer,
                each of shape ``(num_tokens, topk)``.  Order must match
                instantiation order.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"[R3] Number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match number of router instances "
                f"({len(RouterReplay.router_instances)})."
            )
        for i, inst in enumerate(RouterReplay.router_instances):
            inst.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        """Collect recorded topk indices from all instances."""
        return [r.get_recorded_indices() for r in RouterReplay.router_instances]

    @staticmethod
    def clear_global_indices() -> None:
        """Clear recorded and target indices on all instances."""
        for r in RouterReplay.router_instances:
            r.clear_indices()

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction) -> None:
        """Set the replay action for all router instances."""
        for r in RouterReplay.router_instances:
            r.set_router_replay_action(action)

    @staticmethod
    def clear_global_router_replay_action() -> None:
        """Clear the replay action on all router instances."""
        for r in RouterReplay.router_instances:
            r.clear_router_replay_action()

    def __init__(self) -> None:
        self.target_topk_idx: torch.Tensor | None = None
        self.recorded_topk_idx: torch.Tensor | None = None
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_list: list[torch.Tensor] = []
        # 1-D bool mask (shape=(num_tokens,)) marking which
        # rows of ``target_topk_idx`` correspond to real tokens.  Padded
        # rows (seq-alignment slack + batch padding) must not be forced to
        # the recorded top-k (which is [0,...,0]); instead we let them fall
        # back to the live router output so they produce no replay signal.
        self.target_valid_mask: torch.Tensor | None = None
        self.replay_backward_mask_list: list[torch.Tensor | None] = []
        # Keep push_meta queue in sync with backward_list.
        self.replay_push_meta_list: list[dict] = []
        self.creation_order: int = len(RouterReplay.router_instances)
        try:
            import torch.distributed as _dist

            self.creator_rank: int = _dist.get_rank() if _dist.is_initialized() else -1
        except Exception:
            self.creator_rank = -1
        # ------------------------------------------------------------
        RouterReplay.router_instances.append(self)

    def set_target_indices(
        self,
        topk_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Sets the target topk indices (and optional row-validity mask) for replay.

        Args:
            topk_indices: ``(num_tokens, topk)`` replay indices.
            valid_mask: Optional ``(num_tokens,)`` bool tensor. ``True`` means
                the row is a real token and replay should override live routing;
                ``False`` means the row is padding (batch or TP-alignment slack)
                and replay MUST fall back to live routing to avoid forcing those
                rows to expert 0. When ``None``, all rows are treated as real
                (legacy behaviour).
        """
        self.target_topk_idx = topk_indices
        self.target_valid_mask = valid_mask
        self.replay_backward_list.append(topk_indices)
        self.replay_backward_mask_list.append(valid_mask)
        self.replay_push_meta_list.append(None)

    def get_recorded_indices(self) -> torch.Tensor | None:
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        self.recorded_topk_idx = topk_indices

    def clear_indices(self) -> None:
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.target_valid_mask = None
        self.replay_backward_list = []
        self.replay_backward_mask_list = []
        self.replay_push_meta_list = []

    def set_router_replay_action(self, action: RouterReplayAction) -> None:
        self.router_replay_action = action

    def clear_router_replay_action(self) -> None:
        self.router_replay_action = None


# ===================================================================
# Patched routing implementation
# ===================================================================


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    fused: bool,
    router_replay: RouterReplay | None,
    scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched ``topk_routing_with_score_function`` supporting router replay."""
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk and group_limited_topk is not None:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        routing_action = (
            router_replay.router_replay_action if router_replay is not None else None
        )

        if routing_action is None:
            return _compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )

        if routing_action == RouterReplayAction.RECORD:
            probs, top_indices = _compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            if router_replay is not None:
                router_replay.record_indices(top_indices)
            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_FORWARD:
            if router_replay is None or router_replay.target_topk_idx is None:
                # Fallback if replay data is not available
                logger.warning(
                    "[R3] REPLAY_FORWARD: no replay indices available, "
                    "falling back to normal routing."
                )
                return _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )

            # Use the provided indices for replay
            top_indices = router_replay.target_topk_idx
            top_indices = top_indices.to(scores.device)
            # Splice padding rows with live routing to avoid routing to expert 0.
            valid_mask = getattr(router_replay, "target_valid_mask", None)
            if valid_mask is not None and valid_mask.shape[0] == top_indices.shape[0]:
                _, live_top = _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                top_indices = torch.where(
                    valid_mask.to(scores.device).unsqueeze(-1),
                    top_indices,
                    live_top,
                )
            probs = scores.gather(1, top_indices)

            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
            if router_replay is None or not router_replay.replay_backward_list:
                # Fallback if replay data is not available
                logger.warning(
                    "[R3] REPLAY_BACKWARD: no backward indices available, "
                    "falling back to normal routing."
                )
                return _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
            # Backward recompute: use the recorded indices from the forward pass.
            top_indices = router_replay.replay_backward_list.pop(0)
            top_indices = top_indices.to(scores.device)
            bw_mask_list = getattr(router_replay, "replay_backward_mask_list", None)
            if bw_mask_list:
                bw_valid_mask = bw_mask_list.pop(0)
            else:
                bw_valid_mask = None
            # Keep push_meta queue in sync with backward_list.
            _push_meta_list = getattr(router_replay, "replay_push_meta_list", None)
            if _push_meta_list:
                try:
                    _push_meta_list.pop(0)
                except IndexError:
                    pass
            if (
                bw_valid_mask is not None
                and bw_valid_mask.shape[0] == top_indices.shape[0]
            ):
                _, live_top = _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                top_indices = torch.where(
                    bw_valid_mask.to(scores.device).unsqueeze(-1),
                    top_indices,
                    live_top,
                )
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        else:
            return _compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )

    # --- Score function dispatch ---
    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(
                scores_for_routing, topk, num_groups, group_topk
            )
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
        )
    else:
        raise ValueError(f"[R3] Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices),
            torch.ones_like(probs, dtype=routing_map.dtype),
            accumulate=False,
        )
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


# ===================================================================
# Aux-loss helpers
# ===================================================================


def _get_aux_loss_coeff(_self, aux_loss_type: str) -> float:
    """Return the aux loss coeff for the given auxiliary loss type."""
    if isinstance(_self.routing_type, str):
        if _self.routing_type == aux_loss_type:
            return _self.config.moe_aux_loss_coeff
    if isinstance(_self.routing_type, list):
        try:
            idx = _self.routing_type.index(aux_loss_type)
            return _self.config.moe_aux_loss_coeff[idx]
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _is_aux_loss_enabled(_self) -> bool:
    """Check if any auxiliary loss is enabled."""
    for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
        if _get_aux_loss_coeff(_self, aux_loss_type) > 0:
            return True
    return False


# ===================================================================
# patched_routing -- replaces TopKRouter.routing
# ===================================================================


def patched_routing(self, logits: torch.Tensor, *args, **kwargs):
    """Patched ``TopKRouter.routing`` that supports router replay.

    Drop-in replacement for ``TopKRouter.routing`` that delegates to
    ``_patched_topk_routing_with_score_function`` which honours the
    ``RouterReplayAction`` set on the per-layer ``RouterReplay`` instance.
    """
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    moe_router_fusion = getattr(self.config, "moe_router_fusion", False)

    # Calculate probs and routing_map for token dispatching
    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=moe_router_fusion,
            router_replay=getattr(self, "router_replay", None),
        )

    # Apply token dropping to probs and routing_map.
    if (
        self.config.moe_expert_capacity_factor is not None
        and apply_router_token_dropping is not None
    ):
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if not hasattr(self, "is_aux_loss_enabled"):
        self.is_aux_loss_enabled = types.MethodType(_is_aux_loss_enabled, self)

    # Apply aux loss
    if (
        self.training
        and torch.is_grad_enabled()
        and self.is_aux_loss_enabled()
        and compute_routing_scores_for_aux_loss is not None
    ):
        routing_map_for_aux_loss, scores_for_aux_loss = (
            compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion,
            )
        )
        probs = self._apply_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss
        )
        probs = self._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
        )
        probs = self._apply_global_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss
        )

    # Update expert bias and tokens_per_expert
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


# ===================================================================
# Sentinel to prevent double-patching
# ===================================================================
_PATCHES_APPLIED = False

# Store original methods for undo
_ORIGINAL_TF_CONFIG_INIT = None
_ORIGINAL_TOPK_ROUTER_INIT = None
_ORIGINAL_TOPK_ROUTER_ROUTING = None
_ORIGINAL_DISPATCHER_PREPROCESS = None


# ===================================================================
# apply_router_replay_patch
# ===================================================================


def apply_router_replay_patch() -> None:
    """Apply all Megatron-Core monkey-patches required for Router Replay.

    Safe to call multiple times -- subsequent calls are no-ops.
    Must be called **before** model creation.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        logger.debug("[R3] Router replay patches already applied; skipping.")
        return

    logger.info("[R3] Applying Router Replay patches...")

    RouterReplay.router_instances.clear()

    _patch_transformer_config_init()
    _patch_topk_router_init()
    _patch_topk_router_routing()
    _patch_alltoall_dispatcher_preprocess()

    _PATCHES_APPLIED = True
    logger.debug("[R3] All Router Replay patches applied successfully.")


def remove_router_replay_patch() -> None:
    """Undo all patches (primarily for test cleanup)."""
    global _PATCHES_APPLIED
    _undo_transformer_config_patch()
    _undo_topk_router_init_patch()
    _undo_topk_router_routing_patch()
    _undo_dispatcher_patch()
    RouterReplay.router_instances.clear()
    _PATCHES_APPLIED = False
    logger.debug("[R3] All Router Replay patches removed.")


# ===================================================================
# Patch 1: TransformerConfig.__init__
# ===================================================================


def _patch_transformer_config_init() -> None:
    """Patch ``TransformerConfig.__init__`` to accept ``enable_routing_replay``."""
    global _ORIGINAL_TF_CONFIG_INIT

    if getattr(TransformerConfig, "_r3_config_patched", False):
        return

    # Inspect the current signature to add enable_routing_replay
    try:
        sig = inspect.signature(TransformerConfig.__init__)
        native_params = sig.parameters
        params = list(sig.parameters.values())
    except Exception:
        sig = None
        native_params = {}
        params = []

    ext_attr = "enable_routing_replay"

    if ext_attr not in native_params and sig is not None:
        new_param = inspect.Parameter(
            ext_attr, inspect.Parameter.KEYWORD_ONLY, default=False
        )
        if params and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
            params.insert(-1, new_param)
        else:
            params.append(new_param)
        try:
            TransformerConfig.__init__.__signature__ = sig.replace(parameters=params)
        except Exception as e:
            logger.warning("[R3] Failed to update TransformerConfig signature: %s", e)

    _ORIGINAL_TF_CONFIG_INIT = TransformerConfig.__init__

    @wraps(_ORIGINAL_TF_CONFIG_INIT)
    def patched_tf_config_init(self, *args, **kwargs):
        enable_routing_replay = kwargs.get("enable_routing_replay", False)
        if "enable_routing_replay" not in native_params:
            enable_routing_replay = kwargs.pop("enable_routing_replay", False)
        _ORIGINAL_TF_CONFIG_INIT(self, *args, **kwargs)
        self.enable_routing_replay = enable_routing_replay

    TransformerConfig.__init__ = patched_tf_config_init
    TransformerConfig._r3_config_patched = True
    logger.debug(
        "[R3] TransformerConfig.__init__ patched to accept enable_routing_replay."
    )


def _undo_transformer_config_patch() -> None:
    global _ORIGINAL_TF_CONFIG_INIT
    if _ORIGINAL_TF_CONFIG_INIT is not None:
        TransformerConfig.__init__ = _ORIGINAL_TF_CONFIG_INIT
        if hasattr(TransformerConfig, "_r3_config_patched"):
            del TransformerConfig._r3_config_patched
        _ORIGINAL_TF_CONFIG_INIT = None


# ===================================================================
# Patch 2: TopKRouter.__init__
# ===================================================================


def _patch_topk_router_init() -> None:
    """Patch ``TopKRouter.__init__`` to create a ``RouterReplay`` instance."""
    global _ORIGINAL_TOPK_ROUTER_INIT

    if getattr(TopKRouter, "_r3_init_patched", False):
        return

    _ORIGINAL_TOPK_ROUTER_INIT = TopKRouter.__init__

    def patched_init(self, *args, **kwargs):
        _ORIGINAL_TOPK_ROUTER_INIT(self, *args, **kwargs)
        self.router_replay = None
        if getattr(self.config, "enable_routing_replay", False):
            self.router_replay = RouterReplay()
            logger.debug(
                "[R3] TopKRouter: created RouterReplay instance (total instances: %d).",
                len(RouterReplay.router_instances),
            )

    TopKRouter.__init__ = patched_init
    TopKRouter._r3_init_patched = True
    logger.debug("[R3] TopKRouter.__init__ patched.")


def _undo_topk_router_init_patch() -> None:
    global _ORIGINAL_TOPK_ROUTER_INIT
    if _ORIGINAL_TOPK_ROUTER_INIT is not None:
        TopKRouter.__init__ = _ORIGINAL_TOPK_ROUTER_INIT
        if hasattr(TopKRouter, "_r3_init_patched"):
            del TopKRouter._r3_init_patched
        _ORIGINAL_TOPK_ROUTER_INIT = None


# ===================================================================
# Patch 3: TopKRouter.routing
# ===================================================================


def _patch_topk_router_routing() -> None:
    """Patch ``TopKRouter.routing`` with the replay-aware version."""
    global _ORIGINAL_TOPK_ROUTER_ROUTING

    if getattr(TopKRouter, "_r3_routing_patched", False):
        return

    _ORIGINAL_TOPK_ROUTER_ROUTING = TopKRouter.routing
    TopKRouter.routing = patched_routing
    TopKRouter._r3_routing_patched = True
    logger.debug("[R3] TopKRouter.routing patched.")


def _undo_topk_router_routing_patch() -> None:
    global _ORIGINAL_TOPK_ROUTER_ROUTING
    if _ORIGINAL_TOPK_ROUTER_ROUTING is not None:
        TopKRouter.routing = _ORIGINAL_TOPK_ROUTER_ROUTING
        if hasattr(TopKRouter, "_r3_routing_patched"):
            del TopKRouter._r3_routing_patched
        _ORIGINAL_TOPK_ROUTER_ROUTING = None


# ===================================================================
# Patch 4: MoEAlltoAllTokenDispatcher.preprocess
# ===================================================================


def _patch_alltoall_dispatcher_preprocess() -> None:
    """Patch dispatcher preprocess to handle duplicate indices from replay."""
    global _ORIGINAL_DISPATCHER_PREPROCESS

    if MoEAlltoAllTokenDispatcher is None:
        logger.warning(
            "[R3] Cannot import MoEAlltoAllTokenDispatcher -- "
            "skipping preprocess patch."
        )
        return

    if getattr(MoEAlltoAllTokenDispatcher, "_r3_preprocess_patched", False):
        return

    _ORIGINAL_DISPATCHER_PREPROCESS = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(self, routing_map):
        result = _ORIGINAL_DISPATCHER_PREPROCESS(self, routing_map)
        if (
            getattr(self.config, "enable_routing_replay", False)
            and not self.drop_and_pad
            and self.config.moe_expert_capacity_factor is None
            and not (
                getattr(self.config, "moe_router_padding_for_quantization", None)
                or getattr(self.config, "moe_router_padding_for_fp8", None)
            )
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
    MoEAlltoAllTokenDispatcher._r3_preprocess_patched = True
    logger.debug("[R3] MoEAlltoAllTokenDispatcher.preprocess patched.")


def _undo_dispatcher_patch() -> None:
    global _ORIGINAL_DISPATCHER_PREPROCESS
    if MoEAlltoAllTokenDispatcher is None:
        return
    if _ORIGINAL_DISPATCHER_PREPROCESS is not None:
        MoEAlltoAllTokenDispatcher.preprocess = _ORIGINAL_DISPATCHER_PREPROCESS
        if hasattr(MoEAlltoAllTokenDispatcher, "_r3_preprocess_patched"):
            del MoEAlltoAllTokenDispatcher._r3_preprocess_patched
        _ORIGINAL_DISPATCHER_PREPROCESS = None
