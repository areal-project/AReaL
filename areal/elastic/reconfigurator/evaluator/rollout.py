# SPDX-License-Identifier: Apache-2.0

"""SGLang Rollout cost model, scheduler, single-sided replay and SLO capacity."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from heapq import heapify, heappop, heappush
from itertools import count
from time import perf_counter
from typing import ClassVar
from zlib import crc32

from .contracts import Hardware, InferenceRates, ModelShape, RolloutConfig, ServiceSlice
from .workload import ExecutionConfig, Request


def compute_seconds(hardware: Hardware, flops: float, efficiency: float) -> float:
    return flops / (hardware.bf16_tflops * 1e12 * efficiency)


def memory_seconds(hardware: Hardware, moved_bytes: float, efficiency: float) -> float:
    return moved_bytes / (hardware.hbm_gbps * 1e9 * efficiency)


def inference_linear_flops(
    model: ModelShape, tokens: int, emitted: int, tp: int
) -> float:
    dense = model.attention_parameters_per_layer + model.router_parameters_per_layer
    return (
        2 * tokens * model.layers * (dense + model.active_expert_parameters_per_layer)
        + 2 * emitted * model.head_parameters
    ) / tp


def kv_cache_bytes_per_token(
    model: ModelShape, layers: int, local_kv_heads: float, element_bytes: int
) -> float:
    """Count logical bytes for both key and value vectors of one token."""
    return 2 * layers * local_kv_heads * model.head_dim * element_bytes


def supports_layout(model: ModelShape, config: RolloutConfig) -> bool:
    """KV heads may be sharded across attention ranks or replicated onto them."""
    tp = config.attention_tp
    return (
        model.attention_heads % tp == 0
        and (model.kv_heads % tp == 0 or tp % model.kv_heads == 0)
        and model.experts % config.ep_size == 0
        and config.pp <= model.layers
        and (model.mlp_type != "dense" or config.ep_size == 1)
    )


def memory_per_rank(model: ModelShape, config: RolloutConfig) -> tuple[float, float]:
    """Analytical maximum-stage weight bytes and KV bytes per token per DP group."""
    layers = math.ceil(model.layers / config.pp)
    attention = model.attention_parameters_per_rank(config.attention_tp)
    expert = model.expert_parameters_per_layer / config.tp
    replicated = (
        model.router_parameters_per_layer + 2 * model.hidden_size + 2 * model.head_dim
    )
    heads = 1 if config.pp > 1 or model.tied_embeddings else 2
    weights = (
        layers * (attention + expert + replicated)
        + heads * model.head_parameters / config.tp
    ) * config.weight_bytes
    kv = kv_cache_bytes_per_token(
        model,
        layers,
        max(1, model.kv_heads / config.attention_tp),
        config.kv_bytes,
    )
    return weights, kv


def iteration_cost(
    model: ModelShape,
    hardware: Hardware,
    rates: InferenceRates,
    config: RolloutConfig,
    slices: tuple[ServiceSlice, ...],
    kv_after: int,
    *,
    _decode_lines: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    if (
        config.pp != 1
        or config.dp_size != 1
        or config.ep_size != 1
        or config.tp > model.kv_heads
    ):
        return _distributed_iteration_cost(
            model, hardware, rates, config, slices, kv_after, _decode_lines
        )
    m, h, r, tp = model, hardware, rates, config.tp
    tokens = sum(s.query_tokens for s in slices)
    pairs = sum(s.attention_pairs for s in slices)
    pair_flops = m.attention_flops_per_pair(m.layers)
    dense = m.attention_parameters_per_layer + m.router_parameters_per_layer
    emitted = sum(s.emit_token for s in slices)
    linear_flops = inference_linear_flops(m, tokens, emitted, tp)
    attention_flops = pair_flops * pairs / tp
    compute = compute_seconds(h, linear_flops, r.dense) + compute_seconds(
        h, attention_flops, r.attention
    )
    touched = 1 - (1 - m.top_k / m.experts) ** tokens
    weights = (
        (
            m.layers * (dense + touched * m.expert_parameters_per_layer)
            + m.head_parameters
        )
        * config.weight_bytes
        / tp
    )
    local_kv_heads = max(1.0, m.kv_heads / tp)
    kv_width = kv_cache_bytes_per_token(m, m.layers, local_kv_heads, config.kv_bytes)
    kv_traffic = (sum(s.context_tokens for s in slices) + tokens) * kv_width
    activations = tokens * m.layers * m.hidden_size * config.weight_bytes * 24 / tp
    memory = memory_seconds(h, weights + kv_traffic + activations, r.hbm)
    network = 0.0
    if tp > 1:
        remote = tp > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        payload = tokens * m.hidden_size * config.weight_bytes
        network = (
            m.layers
            * r.collectives_per_layer
            * (2 * (tp - 1) / tp * payload / bandwidth + (tp - 1) * latency)
        )
    seconds = max(compute, memory) + network + m.layers * r.launch_seconds
    if _decode_lines is not None:
        fixed = network + m.layers * r.launch_seconds
        _decode_lines.extend(
            (
                (
                    compute + fixed,
                    pair_flops
                    * len(slices)
                    / tp
                    / (h.bf16_tflops * 1e12 * r.attention),
                ),
                (
                    memory + fixed,
                    len(slices) * kv_width / (h.hbm_gbps * 1e9 * r.hbm),
                ),
            )
        )
    lower_bound = m.total_parameters * config.weight_bytes / tp + kv_after * kv_width
    return seconds, lower_bound


def _distributed_iteration_cost(
    model, hardware, rates, config, slices, kv_after, decode_lines=None
):
    """SGLang PP/DP-Attention/EP roofline prior with explicit grouping costs."""
    m, h, r, c = model, hardware, rates, config
    groups: list[list[ServiceSlice]] = [[] for _ in range(c.dp_size)]
    for item in slices:
        groups[SGLangScheduler.attention_group(item.request_id, c.dp_size)].append(item)
    group_tokens = [sum(s.query_tokens for s in row) for row in groups]
    actual_tokens = sum(group_tokens)
    tokens = (
        actual_tokens
        if any(s.kind == "prefill" for s in slices)
        else max(group_tokens) * c.dp_size
    )
    pairs = max(sum(s.attention_pairs for s in row) for row in groups)
    emitted = sum(s.emit_token for s in slices)
    active = m.active_expert_parameters_per_layer
    linear = (
        2
        * m.layers
        * (
            tokens * (active + m.router_parameters_per_layer) / c.tp
            + max(group_tokens) * m.attention_parameters_per_rank(c.attention_tp)
        )
        + 2 * emitted * m.head_parameters / c.tp
    )
    pair_flops = m.attention_flops_per_pair(m.layers)
    attention = pair_flops * pairs / c.attention_tp
    compute = compute_seconds(h, linear, r.dense) + compute_seconds(
        h, attention, r.attention
    )
    weights, kv_width = memory_per_rank(m, c)
    microbatches = min(
        c.pp,
        max(
            1,
            math.ceil(len(slices) / max(1, c.max_batch_sequences // c.dp_size // c.pp)),
        ),
    )
    memory = memory_seconds(
        h,
        weights * microbatches * c.pp
        + (sum(s.context_tokens for s in slices) + actual_tokens)
        * kv_width
        * c.pp
        / c.dp_size
        + tokens * m.layers * m.hidden_size * c.weight_bytes * 24 / c.tp,
        r.hbm,
    )

    def collective(payload, size, span, *, allreduce=True):
        if size == 1:
            return 0.0
        remote = span > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        return (2 if allreduce else 1) * (size - 1) / size * payload / bandwidth + (
            size - 1
        ) * latency

    payload = tokens * m.hidden_size * c.weight_bytes
    network = m.layers * collective(payload / c.dp_size, c.attention_tp, c.attention_tp)
    if m.mlp_type == "dense":
        network += m.layers * collective(payload, c.tp, c.tp)
    else:
        network += (
            2
            * m.layers
            * collective(payload * m.top_k / c.tp, c.ep_size, c.tp, allreduce=False)
        )
        network += m.layers * collective(
            payload * m.top_k / c.ep_size, c.tp // c.ep_size, c.tp
        )
    if c.dp_size > 1:
        network += 2 * m.layers * collective(payload, c.dp_size, c.tp, allreduce=False)
    pipeline = 0.0
    if c.pp > 1:
        remote = c.tp * c.pp > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        pipeline = (c.pp - 1) * (payload / c.tp / bandwidth + microbatches * latency)
    bubble = (c.pp + microbatches - 1) / (c.pp * microbatches)
    seconds = (
        max(compute, memory) + network + m.layers * r.launch_seconds * microbatches
    ) * bubble + pipeline
    if decode_lines is not None:
        fixed = (
            network + m.layers * r.launch_seconds * microbatches
        ) * bubble + pipeline
        for row in groups:
            group_pairs = sum(s.context_tokens + 1 for s in row)
            group_attention = pair_flops * group_pairs / c.attention_tp
            group_compute = compute_seconds(h, linear, r.dense) + compute_seconds(
                h, group_attention, r.attention
            )
            decode_lines.append(
                (
                    group_compute * bubble + fixed,
                    pair_flops
                    * len(row)
                    / c.attention_tp
                    / (h.bf16_tflops * 1e12 * r.attention)
                    * bubble,
                )
            )
        decode_lines.append(
            (
                memory * bubble + fixed,
                len(slices)
                * kv_width
                * c.pp
                / c.dp_size
                / (h.hbm_gbps * 1e9 * r.hbm)
                * bubble,
            )
        )
    return seconds, weights + kv_after / c.dp_size * kv_width


@dataclass(frozen=True)
class ExpertRolloutModel:
    model: ModelShape
    hardware: Hardware
    rates: InferenceRates
    runtime_revision: str

    def __post_init__(self) -> None:
        if not self.runtime_revision:
            raise ValueError("runtime_revision is required")

    def step_seconds(
        self,
        config: RolloutConfig,
        kv_tokens: int,
        ready_requests: tuple[ServiceSlice, ...],
    ) -> float:
        slices = tuple(ready_requests)
        if not slices or len({s.request_id for s in slices}) != len(slices):
            raise ValueError("A service iteration needs distinct requests")
        tokens = sum(s.query_tokens for s in slices)
        kv_after = kv_tokens + tokens
        token_overflow = tokens > config.max_batch_tokens
        if config.prefill_chunk_tokens is not None:
            per_group = [0] * config.dp_size
            for item in slices:
                if item.kind == "prefill":
                    group = SGLangScheduler.attention_group(
                        item.request_id, config.dp_size
                    )
                    per_group[group] += item.query_tokens
            token_overflow = max(per_group) > min(
                config.max_batch_tokens, config.prefill_chunk_tokens // config.dp_size
            )
        if (
            len(slices) > config.max_batch_sequences
            or token_overflow
            or kv_after > config.kv_capacity_tokens
        ):
            raise ValueError("Service batch exceeds sequence/token/KV limit")
        if not supports_layout(self.model, config):
            raise ValueError("Model dimensions do not divide the SGLang layout")
        seconds, lower_bound = iteration_cost(
            self.model, self.hardware, self.rates, config, slices, kv_after
        )
        if lower_bound > self.hardware.memory_gb * 1e9 * config.mem_fraction_static:
            raise ValueError("Rollout memory exceeds static HBM budget")
        return seconds

    def decode_interval(self, config, kv_tokens, slices, max_steps):
        """First step duration, per-step growth and length of one stable decode interval."""
        first = self.step_seconds(config, kv_tokens, slices)
        if max_steps == 1:
            return first, 0.0, 1
        width = len(slices)
        kv_after = kv_tokens + width
        lines: list[tuple[float, float]] = []
        args = self.model, self.hardware, self.rates, config, slices
        _, lower = iteration_cost(*args, kv_after, _decode_lines=lines)
        _, next_lower = iteration_cost(*args, kv_after + width)
        room = self.hardware.memory_gb * 1e9 * config.mem_fraction_static - lower
        steps = min(max_steps, (config.kv_capacity_tokens - kv_tokens) // width)
        if next_lower > lower:
            steps = min(steps, 1 + math.floor(max(0.0, room) / (next_lower - lower)))
        base, slope = max(lines)
        for other_base, other_slope in lines:
            if other_slope > slope:
                steps = min(
                    steps,
                    max(1, math.ceil((base - other_base) / (other_slope - slope))),
                )
        return first, slope, steps


def rollout_config(
    template: RolloutConfig,
    rollout_model: ExpertRolloutModel,
    candidate: ExecutionConfig,
) -> RolloutConfig:
    config = replace(
        template,
        replicas=candidate.rollout_replicas,
        tp=candidate.rollout_tp,
        pp=candidate.rollout_pp,
        dp_size=candidate.sglang_dp_size,
        ep_size=1,
        max_batch_sequences=candidate.sglang_max_running_requests
        or template.max_batch_sequences,
        max_batch_tokens=candidate.sglang_max_prefill_tokens,
        mem_fraction_static=candidate.sglang_mem_fraction_static
        if candidate.sglang_mem_fraction_static is not None
        else template.mem_fraction_static,
    )
    model, hardware = rollout_model.model, rollout_model.hardware
    if not supports_layout(model, config):
        raise ValueError("Model dimensions do not divide the SGLang layout")
    weights, kv_bytes = memory_per_rank(model, config)
    available = hardware.memory_gb * 1e9 * config.mem_fraction_static - weights
    capacity = math.floor(available / kv_bytes) * config.dp_size
    if capacity <= 0:
        raise ValueError("SGLang weights exhaust the configured static memory budget")
    return replace(config, kv_capacity_tokens=capacity)


@dataclass(frozen=True)
class SGLangScheduler:
    """Decode-first/full-reservation prior, with optional internal Attention DP."""

    version: ClassVar[str] = "decode_first_full_reservation_v1"

    @staticmethod
    def attention_group(identity: str, dp_size: int) -> int:
        return crc32(identity.encode()) % dp_size

    def choose_replica(self, config: RolloutConfig, request, replicas):
        need = request.request.input_tokens + request.request.output_tokens - 1
        dp_size = config.dp_size
        inner = self.attention_group(
            f"{request.group_id}/{request.request.name}", dp_size
        )

        def fits(group):
            rows = [
                r
                for r in group.rows.values()
                if self.attention_group(f"{r.group_id}/{r.request.name}", dp_size)
                == inner
            ]
            return (
                len(rows) < config.max_batch_sequences // dp_size
                and sum(
                    r.request.input_tokens + r.request.output_tokens - 1 for r in rows
                )
                + need
                <= config.kv_capacity_tokens // dp_size
            )

        candidates = [
            identity
            for identity, group in replicas.items()
            if len(group.rows) < config.max_batch_sequences
            and group.reserved_tokens + need <= config.kv_capacity_tokens
            and (dp_size == 1 or fits(group))
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda identity: (
                replicas[identity].kv_tokens,
                len(replicas[identity].rows),
                identity,
            ),
        )

    def batch(
        self, config: RolloutConfig, rows
    ) -> tuple[tuple[ServiceSlice, ...], tuple[tuple[str, str], ...]]:
        budget = config.max_batch_tokens
        dp_size = config.dp_size
        chunk = config.prefill_chunk_tokens
        prefill_budgets = (
            [min(budget, chunk // dp_size)] * dp_size if chunk is not None else None
        )
        separate = config.pp > 1
        decode_only = separate and any(r.generated for r in rows)
        slices, keys = [], []
        for request in sorted(rows, key=lambda r: (not bool(r.generated), r.order)):
            if separate and bool(request.generated) != decode_only:
                continue
            if prefill_budgets is None and not budget:
                break
            identity = f"{request.group_id}/{request.request.name}"
            inner = self.attention_group(identity, dp_size)
            if request.generated:
                item = ServiceSlice(identity, request.kv_tokens, 1, "decode", True)
            else:
                remaining = request.request.input_tokens - request.prefilled
                available = (
                    prefill_budgets[inner] if prefill_budgets is not None else budget
                )
                query = min(available, remaining)
                if not query:
                    continue
                item = ServiceSlice(
                    identity, request.prefilled, query, "prefill", query == remaining
                )
            slices.append(item)
            keys.append((request.group_id, request.request.name))
            if prefill_budgets is None:
                budget -= item.query_tokens
            elif not request.generated:
                prefill_budgets[inner] -= item.query_tokens
        return tuple(slices), tuple(keys)


_SGLANG = SGLangScheduler()


@dataclass
class _Call:
    group_id: str
    index: int
    order: int
    ready_at: float
    request: Request
    prefilled: int = 0
    generated: int = 0
    first: float | None = None

    @property
    def kv_tokens(self) -> int:
        return self.prefilled + max(0, self.generated - 1)


@dataclass
class _DecodeSpan:
    first_end: float
    first_seconds: float
    increment: float
    limit: int
    resident_count: int = 0

    def end(self, steps: int) -> float:
        return (
            self.first_end
            + (steps - 1) * self.first_seconds
            + self.increment * steps * (steps - 1) / 2
        )

    def steps_before(self, boundary: float) -> int:
        low, high = 0, self.limit - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self.end(middle) < boundary - 1e-9:
                low = middle
            else:
                high = middle - 1
        return low


@dataclass
class _Replica:
    rows: dict = field(default_factory=dict)
    busy: bool = False
    decode_span: _DecodeSpan | None = None

    @property
    def kv_tokens(self) -> int:
        return sum(r.kv_tokens for r in self.rows.values())

    @property
    def reserved_tokens(self) -> int:
        return sum(
            r.request.input_tokens + r.request.output_tokens - 1
            for r in self.rows.values()
        )


def offered_fraction(forecast, rc, rollout_model, scheduler, slo, concurrency) -> float:
    """Score the first model call of every offered episode, including queued offers."""
    shapes = {e.episode_id: e.turns for e in forecast.episodes}
    offered = {e.episode_id: e.available_at for e in forecast.episodes}
    end = forecast.input_closes_at
    arrivals, waiting, active = deque(), {}, set()
    replicas = {i: _Replica() for i in range(rc.replicas)}
    events, serial = [], count(len(offered))
    passed = 0
    routing_dirty = True

    def push(at, kind, payload=None):
        heappush(events, (at, next(serial), kind, payload))

    def add_call(key, i, at):
        nonlocal routing_dirty
        routing_dirty = True
        turn = shapes[key][i]
        request = Request(f"turn-{i}", turn.input_tokens, turn.output_tokens)
        waiting[key] = _Call(key, i, next(serial), at, request)

    # Stable time order also preserves the original priority of simultaneous offers.
    offers = enumerate(sorted(offered.items(), key=lambda item: item[1]))

    def push_next_offer():
        item = next(offers, None)
        if item is not None:
            index, (key, at) = item
            heappush(events, (at, index, "arrival", (key, at)))

    push_next_offer()
    push(end, "end")

    def simulate_step():
        nonlocal routing_dirty
        while arrivals and len(active) < concurrency:
            key, at = arrivals.popleft()
            active.add(key)
            add_call(key, 0, at)
        if routing_dirty:
            for key, call in tuple(waiting.items()):
                rid = scheduler.choose_replica(rc, call, replicas)
                if rid is not None:
                    replicas[rid].rows[key] = waiting.pop(key)
            routing_dirty = False
        for rid, replica in replicas.items():
            if replica.rows and not replica.busy:
                slices, keys = scheduler.batch(rc, list(replica.rows.values()))
                if not slices:
                    raise ValueError(
                        "Serving profile cannot schedule the pending tokens"
                    )
                replica.decode_span = None
                if all(s.kind == "decode" for s in slices):
                    remaining = min(
                        replica.rows[key].request.output_tokens
                        - replica.rows[key].generated
                        for key, _ in keys
                    )
                    duration, slope, steps = rollout_model.decode_interval(
                        rc, replica.kv_tokens, slices, remaining
                    )
                    if (
                        steps > 1
                        and math.isfinite(slope)
                        and duration + (steps - 1) * slope > 0
                    ):
                        replica.decode_span = _DecodeSpan(
                            now + duration,
                            duration,
                            slope,
                            steps,
                            resident_count=len(replica.rows),
                        )
                else:
                    duration = rollout_model.step_seconds(
                        rc, replica.kv_tokens, tuple(slices)
                    )
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Unknown inference duration")
                replica.busy = True
                push(now + duration, "tokens", (rid, tuple(zip(keys, slices))))

    def advance_decode():
        boundary = end
        spans = []
        for index, (when, _, kind, payload) in enumerate(events):
            if kind == "tokens":
                rid, items = payload
                replica = replicas[rid]
                span = replica.decode_span
                if span is not None and span.resident_count == len(replica.rows):
                    spans.append((index, rid, span, items))
                    boundary = min(boundary, span.end(span.limit))
                    continue
            boundary = min(boundary, when)
        moved = []
        for index, rid, span, items in spans:
            steps = span.steps_before(boundary)
            if steps:
                for (key, _), _ in items:
                    replicas[rid].rows[key].generated += steps
                moved.append((span.end(steps), rid, index))
                span.first_end = span.end(steps + 1)
                span.first_seconds += steps * span.increment
                span.limit -= steps
        if moved:
            for _, rid, index in sorted(moved):
                payload = events[index][3]
                events[index] = (
                    replicas[rid].decode_span.first_end,
                    next(serial),
                    "tokens",
                    payload,
                )
            heapify(events)

    while events:
        now = events[0][0]
        while events and events[0][0] == now:
            _, _, kind, payload = heappop(events)
            if kind == "arrival":
                arrivals.append(payload)
                push_next_offer()
            elif kind == "tool":
                add_call(*payload, now)
            elif kind == "tokens":
                rid, items = payload
                replica = replicas[rid]
                replica.busy = False
                for (key, _), item in items:
                    call = replica.rows[key]
                    if item.kind == "prefill":
                        call.prefilled += item.query_tokens
                    if not item.emit_token:
                        continue
                    if call.generated == 0:
                        call.first = now
                    call.generated += 1
                    if call.generated != call.request.output_tokens:
                        continue
                    del replica.rows[key]
                    routing_dirty = True
                    if call.index == 0:
                        ttft = call.first - offered[key]
                        tpot = (now - call.first) / max(1, call.generated - 1)
                        passed += ttft <= slo.ttft_seconds and tpot <= slo.tpot_seconds
                    turn = shapes[key][call.index]
                    if turn.tool_seconds is None:
                        active.remove(key)
                    else:
                        push(now + turn.tool_seconds, "tool", (key, call.index + 1))
        if now >= end or not (events or arrivals or active):
            break
        simulate_step()
        advance_decode()
    return passed / len(offered)


@dataclass(frozen=True)
class RolloutCapacity:
    lower: float | None
    upper: float | None
    probes: tuple[tuple[float, float | None], ...]
    seconds: float

    @property
    def rate(self) -> float | None:
        return self.lower if self.lower is not None else self.upper


def probe_capacity(
    check, observed_rate, target, *, max_probes=4, require_observed_rate=False
) -> RolloutCapacity:
    started = perf_counter()
    lower = upper = None
    probes = []
    rate = observed_rate
    for _ in range(max_probes):
        score = check(rate)
        probes.append((rate, score))
        if score is None or (
            require_observed_rate and len(probes) == 1 and score < target
        ):
            break
        if score + 1e-12 >= target:
            lower = rate
        else:
            upper = rate
        if lower is None:
            rate /= 2
        elif upper is None:
            rate *= 2
        else:
            rate = (lower + upper) / 2
    return RolloutCapacity(lower, upper, tuple(probes), perf_counter() - started)


def rollout_slo(pattern, rate, config, model, slo, concurrency) -> float:
    """Offer five concurrency windows of episodes at the requested mean rate."""
    episode_count = 5 * concurrency
    forecast = pattern.forecast(0.0, episode_count / rate, episode_count=episode_count)
    if forecast is None:
        raise ValueError("Capacity estimates require observed shapes for all families")
    forecast = replace(forecast, input_closes_at=forecast.input_closes_at + 75.0)
    return offered_fraction(forecast, config, model, _SGLANG, slo, concurrency)
