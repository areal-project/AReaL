# SPDX-License-Identifier: Apache-2.0

"""MoE routing diagnostics, independent of router auxiliary-loss/bias state."""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


def normalize_expert_loads(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return percentages summing to 100 and max / ideal load (1 is balanced)."""
    counts = counts.to(torch.float64)
    total = counts.sum()
    denominator = total.clamp_min(1)
    return 100 * counts / denominator, counts.max() * counts.numel() / denominator


class MoEMetrics:
    """Collect original model forwards only, excluding eval and layer recompute.

    Register the enclosing model part, outside activation-checkpointed layers.
    The router callback runs eagerly so compiled backward graphs cannot capture
    and replay a metrics mutation. Loads include executed alignment padding and
    count each top-k assignment separately; shared experts are excluded.
    """

    def __init__(self) -> None:
        self.active = False
        self.counts: dict[str, torch.Tensor] = {}
        self._handles = []

    def attach(
        self,
        model: nn.Module,
        routers: Iterable[tuple[str, nn.Module]],
        extract_counts: Callable[[Any], torch.Tensor],
    ) -> None:
        routers = list(routers)
        if not routers:
            return
        in_forward = False

        def enter(module: nn.Module, args: tuple[Any, ...]) -> None:
            nonlocal in_forward
            in_forward = True

        def leave(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            nonlocal in_forward
            in_forward = False

        self._handles.append(model.register_forward_pre_hook(enter))
        self._handles.append(model.register_forward_hook(leave, always_call=True))

        def make_hook(layer: str) -> Callable:
            @torch.compiler.disable
            def record(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                if self.active and in_forward:
                    counts = extract_counts(output).detach().to(torch.int64)
                    if layer not in self.counts:
                        self.counts[layer] = counts.clone()
                    else:
                        self.counts[layer].add_(counts)

            return record

        for layer, router in routers:
            self._handles.append(router.register_forward_hook(make_hook(layer)))

    def attach_buffers(
        self, model: nn.Module, layers: Iterable[tuple[str, nn.Module]]
    ) -> None:
        """Read last-forward routing buffers outside compiled/checkpointed layers.

        Recompute may overwrite these buffers but never invokes the enclosing
        model's hook. This keeps all metrics accumulation outside fullgraph code.
        """
        layers = list(layers)
        if not layers:
            return

        def record(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if self.active:
                for layer, moe in layers:
                    counts = moe.routing_counts.detach()
                    if layer not in self.counts:
                        self.counts[layer] = counts.clone()
                    else:
                        self.counts[layer].add_(counts)

        self._handles.append(model.register_forward_hook(record))

    @contextmanager
    def measure(self) -> Iterator[None]:
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def export(
        self,
        *,
        reduce_group: dist.ProcessGroup | None,
        pp_group: dist.ProcessGroup | None = None,
        replicas: int = 1,
    ) -> dict[str, float]:
        result = {}
        if self.counts:
            layers = sorted(self.counts)
            sizes = [self.counts[layer].numel() for layer in layers]
            packed = torch.cat([self.counts[layer] for layer in layers])
            if dist.is_initialized():
                dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=reduce_group)
            for layer, counts in zip(
                layers,
                (packed.cpu().to(torch.float64) / replicas).split(sizes),
                strict=True,
            ):
                percentages, ratio = normalize_expert_loads(counts)
                prefix = f"moe_balance/layer_{layer}"
                result[f"{prefix}/max_over_ideal"] = ratio.item()
                for expert, (count, percent) in enumerate(
                    zip(counts.tolist(), percentages.tolist(), strict=True)
                ):
                    result[f"{prefix}/expert_{expert}/tokens"] = count
                    result[f"{prefix}/expert_{expert}/load_percent"] = percent
        if pp_group is not None and dist.is_initialized():
            stages = [None] * dist.get_world_size(pp_group)
            dist.all_gather_object(stages, result, group=pp_group)
            result = {key: value for stage in stages for key, value in stage.items()}
        self.counts.clear()
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def split_expert_load_metrics(
    data: dict[str, float],
) -> tuple[dict[str, float], list[list[float]]]:
    """Separate per-expert rows from scalar metrics for W&B Table logging.

    Rows are [layer, expert, tokens, load_percent], sorted numerically. Per-layer
    max_over_ideal remains a scalar; no model-sized family of W&B scalar keys is
    created for the expert matrix.
    """
    import re

    pattern = re.compile(
        r"^moe_balance/layer_(\d+)/expert_(\d+)/(tokens|load_percent)$"
    )
    scalars = {}
    rows = {}
    for key, value in data.items():
        match = pattern.match(key)
        if match is None:
            scalars[key] = value
            continue
        layer, expert = int(match[1]), int(match[2])
        row = rows.setdefault((layer, expert), [layer, expert, 0.0, 0.0])
        row[2 if match[3] == "tokens" else 3] = value
    return scalars, [rows[key] for key in sorted(rows)]
