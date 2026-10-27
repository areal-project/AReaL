# SPDX-License-Identifier: Apache-2.0

"""Keep generated visual special tokens in the text domain of native Bridge."""

import inspect
from functools import wraps
from types import FunctionType, MethodType
from typing import Any

import torch


def visual_lookup_ids(
    input_ids: torch.Tensor, token_types: torch.Tensor, model: Any
) -> torch.Tensor:
    """Build IDs used only by visual scatter and mRoPE, never by embeddings."""
    if input_ids.shape != token_types.shape:
        raise ValueError("Visual token types must match native Bridge input_ids")
    token_types = token_types.to(device=input_ids.device)
    visual = (input_ids == model.image_token_id) | (input_ids == model.video_token_id)
    # Vision-start markers themselves have type zero in HF processors. Keep a
    # marker only if its following token belongs to an actual image/video span.
    start = input_ids == model.vision_start_token_id
    actual_start = torch.zeros_like(start)
    actual_start[..., :-1] = token_types[..., 1:] != 0
    return input_ids.masked_fill(
        (visual & (token_types == 0)) | (start & ~actual_start), -1
    )


def install_qwen3_vl_token_mask(
    models: list[torch.nn.Module],
) -> list[torch.nn.Module]:
    """Bridge pre-wrap hook for Qwen3-VL and Qwen3.5's native VLM forward.

    The pinned forward ignores mm_token_type_ids. Bind its visual utilities to
    this invocation, without changing dependency globals or token embeddings.
    The native forward still owns vision execution, packing, CP and deepstack.
    """
    for model in models:
        if getattr(model, "_areal_qwen3_vl_token_mask", False):
            continue
        original = model.forward.__func__
        namespace = original.__globals__
        required = ("reorganize_inputs", "get_rope_index", "preprocess_packed_seqs")
        if not all(callable(namespace.get(name)) for name in required):
            raise ValueError("Unsupported native Qwen-VL forward utility contract")
        signature = inspect.signature(original)
        reorganize = namespace["reorganize_inputs"]
        rope = namespace["get_rope_index"]
        pack = namespace["preprocess_packed_seqs"]
        rope_signature = inspect.signature(rope)

        # Capture each model's implementation independently (virtual PP chunks
        # may call different forwards). No module-global monkey patches.
        def make_forward(
            original, namespace, signature, reorganize, rope, pack, rope_signature
        ):
            @wraps(original)
            def forward(self, *args, **kwargs):
                arguments = signature.bind(self, *args, **kwargs)
                arguments.apply_defaults()
                values = arguments.arguments
                input_ids = values["input_ids"]
                types = values.get("mm_token_type_ids")
                if types is None:
                    if any(
                        values.get(key) is not None
                        for key in ("pixel_values", "pixel_values_videos")
                    ):
                        raise ValueError(
                            "Native Qwen-VL vision requires mm_token_type_ids"
                        )
                    return original(self, *args, **kwargs)
                lookup_ids = visual_lookup_ids(input_ids, types, self)
                packed_types = None

                def typed_reorganize(*call_args, **call_kwargs):
                    nonlocal packed_types
                    ids = call_kwargs.get(
                        "input_ids", call_args[0] if call_args else None
                    )
                    if ids is input_ids:
                        return reorganize(
                            **{
                                **call_kwargs,
                                "input_ids": lookup_ids,
                                "image_input_mask": lookup_ids == self.image_token_id,
                                "video_input_mask": lookup_ids == self.video_token_id,
                            }
                        )
                    # The native forward's second call only consumes the mask
                    # after packing/CP. Reorganizing full image grids using a
                    # CP-local ID subset would also break mixed image/video.
                    if values.get("packed_seq_params") is None:
                        raise ValueError("Unexpected native Qwen-VL visual ID layout")
                    if packed_types is None:
                        mask = values.get("attention_mask")
                        if mask is None:
                            mask = torch.ones_like(input_ids, dtype=torch.bool)
                        packed_types = pack(
                            types,
                            mask,
                            pre_process=True,
                            pg_collection=self.pg_collection,
                        )[0]
                    if packed_types.shape != ids.shape:
                        raise ValueError(
                            "Packed visual token types must match CP-local IDs"
                        )
                    visual_mask = (
                        (ids == self.image_token_id) | (ids == self.video_token_id)
                    ) & (packed_types != 0)
                    return None, None, visual_mask

                def typed_rope(*call_args, **call_kwargs):
                    bound = rope_signature.bind(*call_args, **call_kwargs)
                    bound.arguments["input_ids"] = lookup_ids
                    return rope(*bound.args, **bound.kwargs)

                scoped = FunctionType(
                    original.__code__,
                    {
                        **namespace,
                        "reorganize_inputs": typed_reorganize,
                        "get_rope_index": typed_rope,
                    },
                    original.__name__,
                    original.__defaults__,
                    original.__closure__,
                )
                scoped.__kwdefaults__ = original.__kwdefaults__
                return scoped(self, *args, **kwargs)

            return forward

        model.forward = MethodType(
            make_forward(
                original, namespace, signature, reorganize, rope, pack, rope_signature
            ),
            model,
        )
        model._areal_qwen3_vl_token_mask = True
    return models
