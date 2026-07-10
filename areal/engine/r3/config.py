# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from areal.engine.r3.asserts import r3_error


@dataclass(frozen=True)
class R3MoEConfig:
    num_layers: int
    num_moe_layers: int
    topk: int
    moe_layer_indices: tuple[int, ...]
    global_to_moe_index: dict[int, int]


def _get_required_int(config: Any, names: tuple[str, ...], *, label: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            value = int(value)
            if value <= 0:
                raise r3_error(f"{label} must be positive", field=name, value=value)
            return value
    raise r3_error(f"Unable to resolve {label}", candidate_fields=names)


def resolve_moe_layer_indices(config: Any) -> tuple[int, ...]:
    num_layers = _get_required_int(config, ("num_layers",), label="num_layers")
    freq = getattr(config, "moe_layer_freq", 1)

    if isinstance(freq, int):
        if freq == 0:
            return ()
        step = abs(freq)
        indices = list(range(step - 1, num_layers, step))
    elif isinstance(freq, (list, tuple)):
        if len(freq) != num_layers:
            raise r3_error(
                "moe_layer_freq length must match num_layers",
                moe_layer_freq_len=len(freq),
                num_layers=num_layers,
            )
        indices = [idx for idx, is_moe in enumerate(freq) if is_moe]
    else:
        raise r3_error(
            "Unsupported moe_layer_freq type",
            moe_layer_freq_type=type(freq).__name__,
        )

    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    if first_dense > 0:
        indices = [idx for idx in indices if idx >= first_dense]
    return tuple(indices)


def resolve_r3_moe_config(config: Any) -> R3MoEConfig:
    topk = _get_required_int(
        config,
        ("moe_router_topk", "num_experts_per_tok"),
        label="MoE router topk",
    )
    moe_layer_indices = resolve_moe_layer_indices(config)
    return R3MoEConfig(
        num_layers=_get_required_int(config, ("num_layers",), label="num_layers"),
        num_moe_layers=len(moe_layer_indices),
        topk=topk,
        moe_layer_indices=moe_layer_indices,
        global_to_moe_index={
            global_idx: moe_idx for moe_idx, global_idx in enumerate(moe_layer_indices)
        },
    )
