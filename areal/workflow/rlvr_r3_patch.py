# SPDX-License-Identifier: Apache-2.0

"""R3 helpers for the RLVR workflow.

Bridges inference-time ``ModelResponse.routed_experts`` into the training-side
tensor dict so that the Megatron engine can replay routing decisions.
"""

from __future__ import annotations

import numpy as np
import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/rlvr")

_RESOLVED_CACHE: dict[str, tuple[int, int]] = {}


def resolve_r3_moe_config(model_path: str) -> tuple[int, int]:
    """Resolve ``num_moe_layers`` and ``topk`` from the HuggingFace model config.

    Inspects the model's ``config.json`` for standard MoE fields:

    * ``num_experts_per_tok`` → topk
    * ``num_hidden_layers`` and ``first_k_dense_replace`` → num_moe_layers
      (MoE layers = total layers - first_k_dense_replace)
    * ``moe_layer_freq`` (int or list) → used when available for precise counting

    Results are cached per ``model_path`` to avoid repeated disk reads.

    Args:
        model_path: Path or repo ID of the HuggingFace model.

    Returns:
        ``(num_moe_layers, topk)`` tuple.

    Raises:
        ValueError: If the model config does not contain sufficient MoE fields.
    """
    if model_path in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[model_path]

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    topk = getattr(hf_config, "num_experts_per_tok", None)
    if topk is None:
        raise ValueError(
            "[R3] Cannot resolve topk from model config: "
            f"'num_experts_per_tok' not found in {type(hf_config).__name__} "
            f"at model_path={model_path}. This model may not be a MoE model, "
            "or uses a non-standard config field name for router top-k."
        )

    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            "[R3] Cannot resolve num_moe_layers from model config: "
            f"'num_hidden_layers' not found in {type(hf_config).__name__} "
            f"at model_path={model_path}."
        )

    moe_layer_freq = getattr(hf_config, "moe_layer_freq", None)
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", None)

    if moe_layer_freq is not None:
        if isinstance(moe_layer_freq, int):
            num_moe_layers = sum(
                1 for i in range(num_hidden_layers) if i % moe_layer_freq == 0
            )
        elif isinstance(moe_layer_freq, (list, tuple)):
            num_moe_layers = sum(1 for v in moe_layer_freq if v == 1)
        else:
            raise ValueError(
                f"[R3] Unsupported moe_layer_freq type: {type(moe_layer_freq)}"
            )
    elif first_k_dense_replace is not None:
        num_moe_layers = num_hidden_layers - first_k_dense_replace
    else:
        num_moe_layers = num_hidden_layers

    if num_moe_layers <= 0:
        raise ValueError(
            "[R3] Resolved num_moe_layers=0 from model config. "
            f"num_hidden_layers={num_hidden_layers}, "
            f"moe_layer_freq={moe_layer_freq}, "
            f"first_k_dense_replace={first_k_dense_replace}. "
            "This model may not be a MoE model."
        )

    _RESOLVED_CACHE[model_path] = (num_moe_layers, topk)
    logger.info(
        "[R3] Resolved from model config at %s: "
        "num_moe_layers=%d, topk=%d "
        "(num_hidden_layers=%d, moe_layer_freq=%s, first_k_dense_replace=%s).",
        model_path,
        num_moe_layers,
        topk,
        num_hidden_layers,
        moe_layer_freq,
        first_k_dense_replace,
    )
    return num_moe_layers, topk


def extract_routed_experts(
    routed_experts_np: np.ndarray | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_moe_layers: int | None = None,
    topk: int | None = None,
    compress_dtype: bool = True,
) -> torch.Tensor | None:
    """Convert ``ModelResponse.routed_experts`` into a training tensor.

    Args:
        routed_experts_np: ``np.ndarray`` of shape ``(num_sgl_tokens, num_moe_layers * topk)``
            as returned by the SGLang inference backend, or ``None``.
        input_ids: ``(1, seq_len)`` token ids (prompt + response).
        attention_mask: ``(1, seq_len)`` with 1 for real tokens, 0 for padding.
        num_moe_layers: Number of MoE layers. Required -- resolved from model config.
        topk: Router top-k. Required -- resolved from model config.
        compress_dtype: Downcast to ``uint8`` / ``int16`` when possible.

    Returns:
        ``torch.Tensor`` of shape ``(1, seq_len, num_moe_layers, topk)`` or ``None``.

    Raises:
        ValueError: If ``num_moe_layers`` or ``topk`` is not provided.
    """
    if routed_experts_np is None:
        return None

    if num_moe_layers is None or topk is None:
        raise ValueError(
            "[R3] num_moe_layers and topk are required for routed_experts "
            "preprocessing. These should be resolved from the model config "
            "via resolve_r3_moe_config(model_path). Shape-based inference "
            f"(decomposing flat_dim={routed_experts_np.shape[1]}) is "
            "ambiguous and can silently corrupt training."
        )

    try:
        from areal.engine.router_replay_utils import (
            preprocess_routed_experts_batch,
        )

        result = preprocess_routed_experts_batch(
            routed_experts_np,
            input_ids,
            attention_mask,
            num_moe_layers=num_moe_layers,
            topk=topk,
            compress_dtype=compress_dtype,
        )
        return result
    except Exception:
        logger.warning(
            "[R3] Failed to preprocess routed_experts (shape=%s); skipping.",
            getattr(routed_experts_np, "shape", "unknown"),
            exc_info=True,
        )
        return None


def inject_routed_experts_into_result(
    result: dict[str, torch.Tensor],
    routed_experts: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Add ``routed_experts`` to the result dict if available.

    This is a trivial helper kept separate for clarity and to centralise
    the key name (``"routed_experts"``).
    """
    if routed_experts is not None:
        result["routed_experts"] = routed_experts
    return result
