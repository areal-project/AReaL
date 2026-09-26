# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from typing import Any

import torch

_VISION_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
)


def require_qwen4_exp_vision_runtime() -> type[Any]:
    """Fail before model allocation if the runtime cannot construct vision mRoPE."""
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen4Exp vision requires a Transformers runtime with "
            "Qwen4ExpModel.get_rope_index and get_vision_position_ids "
            "(validated with Transformers 5.16.1). The standard locked "
            "5.3.0/5.7.0 environments do not support this model."
        ) from exc
    if not all(
        callable(getattr(Qwen4ExpModel, name, None))
        for name in ("get_rope_index", "get_vision_position_ids")
    ):
        raise RuntimeError("Qwen4Exp vision runtime lacks required mRoPE methods.")
    return Qwen4ExpModel


def install_qwen4_exp_visual_token_mask(model: torch.nn.Module) -> None:
    """Scope the pinned bridge's ID-only visual scatter to actual visual tokens.

    The bridge calls this after token embedding lookup and CP ID reconstruction.
    A private ID copy masks text occurrences only for visual scatter; the model's
    original IDs, labels, PLE inputs and text embeddings are never replaced.
    """
    visual = getattr(model, "visual", None)
    if visual is None or getattr(visual, "_areal_visual_token_mask", False):
        return
    original = visual.get_inputs_embeds

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        token_types = kwargs.get("mm_token_type_ids")
        if token_types is None:
            if any(
                kwargs.get(key) is not None
                for key in ("pixel_values", "pixel_values_videos")
            ):
                raise ValueError("Qwen4Exp visual scatter requires mm_token_type_ids")
            return original(inputs_embeds, **kwargs)
        input_ids = kwargs["input_ids"]
        if token_types.shape != input_ids.shape:
            raise ValueError("Visual token types must match reconstructed input_ids")
        token_types = token_types.to(device=input_ids.device)
        special = (input_ids == self.hf_config.image_token_id) | (
            input_ids == self.hf_config.video_token_id
        )
        # -1 is only consumed by the bridge's equality masks, never an embedding lookup.
        scatter_ids = input_ids.masked_fill(special & (token_types == 0), -1)
        return original(inputs_embeds, **{**kwargs, "input_ids": scatter_ids})

    visual.get_inputs_embeds = MethodType(get_inputs_embeds, visual)
    visual._areal_visual_token_mask = True


def prepare_qwen4_exp_mrope_inputs(
    data: dict[str, Any],
    hf_config: Any,
    processor: Any = None,
    language_model_only: bool = False,
) -> dict[str, Any]:
    """Compute upstream Qwen4Exp mRoPE before AReaL reorders and packs samples.

    Position ids use AReaL's existing ``[B, S, 3]`` multimodal packing layout.
    Only the Hugging Face model's weight-free position methods are bound; no
    vision or language model parameters are constructed.
    """
    result = dict(data)
    input_ids = data["input_ids"].to(torch.long)
    attention_mask = data["attention_mask"].bool()
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "Qwen4Exp mRoPE requires padded [B, S] ids and attention_mask."
        )

    batch_size = input_ids.shape[0]
    samples = data.get("multi_modal_input")
    top_level_vision = {
        key: data[key] for key in _VISION_KEYS if data.get(key) is not None
    }
    if samples is None:
        if top_level_vision and batch_size != 1:
            raise ValueError(
                "Batched Qwen4Exp vision inputs require per-sample multi_modal_input "
                "so pixels and grids follow microbatch reordering."
            )
        samples = (
            [top_level_vision] if batch_size == 1 else [{} for _ in range(batch_size)]
        )
    elif top_level_vision:
        raise ValueError(
            "Qwen4Exp vision inputs must not duplicate nested and top-level payloads."
        )
    if len(samples) != batch_size:
        raise ValueError(
            "multi_modal_input must have one entry per Qwen4Exp batch row."
        )

    vision_tokens = (
        (input_ids == hf_config.image_token_id)
        | (input_ids == hf_config.video_token_id)
    ) & attention_mask
    # Generated special tokens are text, not evidence of image/video payloads.
    # Trainers preserve the input-aligned mask before shifting loss_mask for labels.
    # Raw batches still carry the unshifted mask in loss_mask.
    loss_mask = data.get("input_token_loss_mask", data.get("loss_mask"))
    if loss_mask is not None:
        if loss_mask.shape != input_ids.shape:
            raise ValueError("loss_mask must have the same [B, S] shape as input_ids.")
        vision_tokens = vision_tokens & ~loss_mask.to(
            device=input_ids.device, dtype=torch.bool
        )
    token_types = data.get("mm_token_type_ids")
    has_typed_vision = token_types is not None and bool((token_types != 0).any())
    has_vision = (
        has_typed_vision
        or bool(vision_tokens.any())
        or any(
            item.get(key) is not None
            and (not torch.is_tensor(item[key]) or item[key].numel() > 0)
            for item in samples
            for key in _VISION_KEYS
        )
    )
    if not has_vision:
        return result
    if language_model_only:
        raise ValueError(
            "language_model_only=True cannot consume image or video inputs."
        )

    token_types = data.get("mm_token_type_ids")
    if token_types is None:
        if processor is None or not callable(
            getattr(processor, "create_mm_token_type_ids", None)
        ):
            raise ValueError(
                "Qwen4Exp vision inputs require processor-produced mm_token_type_ids "
                "or a processor with create_mm_token_type_ids()."
            )
        token_types = torch.tensor(
            processor.create_mm_token_type_ids(input_ids.tolist()),
            device=input_ids.device,
            dtype=torch.long,
        )
    if token_types.shape != input_ids.shape:
        raise ValueError(
            "mm_token_type_ids must have the same [B, S] shape as input_ids."
        )
    token_types = token_types.to(device=input_ids.device, dtype=torch.long)
    generated = torch.zeros_like(attention_mask)
    if loss_mask is not None:
        generated = loss_mask.to(device=input_ids.device, dtype=torch.bool)
        token_types = token_types.masked_fill(generated, 0)
    if ((token_types < 0) | (token_types > 2))[attention_mask].any():
        raise ValueError(
            "Qwen4Exp supports text, image, and video modality token types."
        )

    merge_size = hf_config.vision_config.spatial_merge_size
    for modality, token_id, pixel_key, grid_key in (
        (1, hf_config.image_token_id, "pixel_values", "image_grid_thw"),
        (2, hf_config.video_token_id, "pixel_values_videos", "video_grid_thw"),
    ):
        if (
            ((token_types == modality) != ((input_ids == token_id) & ~generated))
            & attention_mask
        ).any():
            raise ValueError(
                f"mm_token_type_ids disagree with Qwen4Exp {grid_key} token ids."
            )
        for index, item in enumerate(samples):
            types = token_types[index][attention_mask[index]]
            token_count = int((types == modality).sum())
            pixels, grid = item.get(pixel_key), item.get(grid_key)
            if pixels is None and grid is None and token_count == 0:
                continue
            if pixels is None or grid is None:
                raise ValueError(
                    f"Sample {index} requires both {pixel_key} and {grid_key}."
                )
            if (
                grid.ndim != 2
                or grid.shape[1] != 3
                or (grid <= 0).any()
                or (grid[:, 1:] % merge_size != 0).any()
            ):
                raise ValueError(f"Sample {index} has an invalid {grid_key} grid.")
            patch_count = int(grid.prod(-1).sum())
            if (
                pixels.shape[0] != patch_count
                or token_count != patch_count // merge_size**2
            ):
                raise ValueError(
                    f"Sample {index} {pixel_key}, {grid_key}, and placeholder counts differ."
                )
            modalities, lengths = torch.unique_consecutive(types, return_counts=True)
            group_lengths = lengths[modalities == modality]
            if modality == 1:
                expected_lengths = grid.prod(-1) // merge_size**2
            else:
                expected_lengths = torch.repeat_interleave(
                    grid[:, 1:].prod(-1) // merge_size**2, grid[:, 0]
                )
            if not torch.equal(group_lengths, expected_lengths.to(types.device)):
                raise ValueError(
                    f"Sample {index} {grid_key} does not match modality token groups."
                )

    Qwen4ExpModel = require_qwen4_exp_vision_runtime()
    context = SimpleNamespace(config=hf_config)
    context.get_vision_position_ids = MethodType(
        Qwen4ExpModel.get_vision_position_ids, context
    )
    grids = {}
    for key in ("image_grid_thw", "video_grid_thw"):
        values = [
            item[key].to(input_ids.device)
            for item in samples
            if item.get(key) is not None
        ]
        grids[key] = torch.cat(values, dim=0) if values else None
    position_ids, _ = Qwen4ExpModel.get_rope_index(
        context,
        input_ids=input_ids,
        mm_token_type_ids=token_types,
        attention_mask=attention_mask,
        **grids,
    )
    for key in top_level_vision:
        result.pop(key)
    result["multi_modal_input"] = [dict(item) for item in samples]
    result["input_ids"] = input_ids
    result["mm_token_type_ids"] = token_types
    result["position_ids"] = position_ids.permute(1, 2, 0).contiguous()
    return result
