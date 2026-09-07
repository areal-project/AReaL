# SPDX-License-Identifier: Apache-2.0

"""Per-tensor helpers for Bailing V3 NCCL weight updates."""

import re

import torch


def is_bailing_v3(hf_config) -> bool:
    # V2.5 and V3 checkpoints both use model_type="bailing_hybrid".
    return (
        "BailingMoeV3ForCausalLM" in (getattr(hf_config, "architectures", None) or [])
        or getattr(hf_config, "model_type", None) == "bailing_moe_v3"
    )


def validate_bailing_v3_weight_update(
    hf_config,
    *,
    use_lora: bool = False,
    quantization_config=None,
    fp8_direct_convert: bool = False,
) -> None:
    """Reject layouts that the V3 per-tensor bridge cannot export."""
    if use_lora or quantization_config or fp8_direct_convert:
        raise NotImplementedError(
            "BailingMoeV3 NCCL weight updates require unquantized full-model "
            "weights; LoRA and FP8 conversion are not supported."
        )
    if not getattr(hf_config, "no_kda_lora", True):
        raise NotImplementedError(
            "BailingMoeV3 NCCL weight updates require no_kda_lora=True; "
            "the bridge exports the fused [q, k, v, f, g] KDA layout."
        )


class BailingV3MlaWeightPairs:
    """Keep MLA down projections in one call to SGLang's ``load_weights``.

    The low-rank Q and KV projections are fused by the inference loader using a
    call-local cache. A bucket boundary between them silently drops both updates.
    Megatron visits each layer's parameters together, so at most one incomplete
    pair is held here, independently of the ordinary byte-limited weight bucket.
    """

    _PATTERN = re.compile(
        r"model\.layers\.(\d+)\.attention\.(q_a_proj|kv_a_proj_with_mqa)\.weight$"
    )

    def __init__(self):
        self._pending: tuple[str, torch.Tensor] | None = None

    def group(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        grouped = []
        for name, tensor in weights:
            match = self._PATTERN.fullmatch(name)
            if match is None:
                grouped.append((name, tensor))
                continue
            if self._pending is None:
                self._pending = (name, tensor)
                continue
            pending_name, pending_tensor = self._pending
            pending_match = self._PATTERN.fullmatch(pending_name)
            if pending_match.group(1) != match.group(1) or pending_match.group(
                2
            ) == match.group(2):
                raise RuntimeError(
                    "BailingMoeV3 MLA weight updates require both down projections "
                    f"of one layer before the next pair: {pending_name}, {name}."
                )
            grouped.extend([(pending_name, pending_tensor), (name, tensor)])
            self._pending = None
        return grouped

    def finish(self) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "Incomplete BailingMoeV3 MLA down-projection pair at the end of "
                f"the weight update: {self._pending[0]}."
            )
