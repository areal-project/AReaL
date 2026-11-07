# SPDX-License-Identifier: Apache-2.0
"""Opt-in batch occupancy and forward-stream timing; no device synchronization."""

from __future__ import annotations

import json
import os
import socket
import time
from collections import deque
from contextlib import contextmanager
from functools import wraps
from numbers import Integral
from pathlib import Path
from typing import Any

from areal.utils import logging

TRACE_DIR_ENV = "AREAL_ROLLOUT_TRACE_DIR"
logger = logging.getLogger("SGLangBatchObserver")


def snapshot_batch(batch: Any) -> dict[str, Any] | None:
    """Snapshot CPU scheduling lengths before overlap can mutate the batch.

    Context counts the full prompt, even during chunked prefill, plus generated
    positions already scheduled. Input counts only positions fed to this forward.
    """
    mode = batch.forward_mode.name.lower()
    if mode not in {"extend", "decode", "mixed"}:
        return None
    lengths = batch.seq_lens_cpu
    if getattr(getattr(lengths, "device", None), "type", "cpu") != "cpu":
        raise ValueError("Batch observer requires CPU sequence lengths")
    lengths = [int(length) for length in lengths]
    if len(lengths) != len(batch.reqs):
        raise ValueError("Batch observer request/sequence lengths do not match")
    decoding = (
        {id(req) for req in (batch.decoding_reqs or [])} if mode == "mixed" else set()
    )
    phases = {
        phase: dict(
            requests=0,
            prompt_tokens=0,
            generated_tokens=0,
            context_tokens=0,
            input_tokens=0,
        )
        for phase in ("prefill", "decode")
    }
    requests = []
    for i, (req, length) in enumerate(zip(batch.reqs, lengths)):
        phase = "decode" if mode == "decode" or id(req) in decoding else "prefill"
        prompt = len(req.origin_input_ids)
        generated = max(0, length - prompt)
        stats = phases[phase]
        stats["requests"] += 1
        stats["prompt_tokens"] += prompt
        stats["generated_tokens"] += generated
        stats["context_tokens"] += prompt + generated
        query = 1 if phase == "decode" else int(batch.extend_lens[i])
        stats["input_tokens"] += query
        past = length - query
        requests.append(
            {
                "request_id": getattr(req, "rid", None),
                "phase": phase,
                "prompt_tokens": prompt,
                "sequence_tokens": length,
                "query_tokens": query,
                "past_tokens": past,
                "scheduled_generated_tokens": generated,
                "attention_pairs": query * past + query * (query + 1) // 2,
            }
        )
    return {
        "forward_mode": mode,
        **phases,
        "requests": requests,
        "attention_pairs": sum(req["attention_pairs"] for req in requests),
    }


class BatchObserver:
    """Write immutable snapshots after their CUDA end event is ready.

    GPU offsets are exact event offsets, not CPU enqueue durations. The mapping
    to wall time is bounded by CPU observations; both bounds are saved. These
    intervals may contain stream waits and are not a kernel-busy profiler.
    """

    def __init__(self, scheduler: Any, directory: str) -> None:
        self.scheduler = scheduler
        self.device = scheduler.device_module
        self.pending: deque = deque()
        self.origin = None
        self.wall_lower = self.wall_upper = None
        self.wall_base, self.mono_base = time.time(), time.perf_counter()
        self.sequence = 0
        self.closed = False
        self.identity = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **{
                name: getattr(scheduler, name, None)
                for name in ("gpu_id", "tp_rank", "pp_rank", "dp_rank", "attn_cp_rank")
            },
        }
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f"batches-{self.identity['host']}-{os.getpid()}.jsonl"
        self.file = self.path.open("x", encoding="utf-8", buffering=1)
        self._write(
            {
                "record": "metadata",
                "schema_version": 2,
                **self.identity,
                "timing": "CUDA events on the observed rank's forward stream",
                "context_tokens": "full prompt plus scheduled generated positions",
            }
        )

    def _wall(self) -> float:
        return self.wall_base + time.perf_counter() - self.mono_base

    def _write(self, row: dict[str, Any]) -> None:
        self.file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _disable(self) -> None:
        logger.exception("Rollout batch collection stopped: %s", self.path)
        self.closed = True
        self.pending.clear()
        try:
            self.file.close()
        except OSError:
            logger.exception("Could not close rollout trace: %s", self.path)

    def start(self, batch: Any) -> tuple | None:
        if self.closed:
            return None
        self.drain()
        if self.closed:
            return None
        try:
            row = snapshot_batch(batch)
            if row is None:
                return None
            capacity = getattr(self.scheduler, "max_total_num_tokens", None)
            allocator = getattr(self.scheduler, "token_to_kv_pool_allocator", None)
            available = allocator.available_size() if allocator is not None else None
            row["kv_capacity_tokens"] = (
                int(capacity) if isinstance(capacity, Integral) else None
            )
            row["kv_allocated_tokens_at_forward_entry"] = (
                int(capacity - available)
                if isinstance(capacity, Integral) and isinstance(available, Integral)
                else None
            )
            if self.origin is None:
                self.origin = self.device.Event(enable_timing=True)
                self.wall_lower = self._wall()
                self.origin.record()
            start, end = (self.device.Event(enable_timing=True) for _ in range(2))
            self.sequence += 1
            row.update(
                record="batch", batch_id=self.sequence, host_started=self._wall()
            )
            start.record()
            return row, start, end
        except Exception:
            self._disable()
            return None

    def finish(self, item: tuple | None) -> None:
        if item is None or self.closed:
            return
        try:
            row, start, end = item
            end.record()
            row["host_finished"] = self._wall()
            self.pending.append((row, start, end))
            self.drain()
        except Exception:
            self._disable()

    def drain(self) -> None:
        if self.closed:
            return
        try:
            while self.pending and self.pending[0][2].query():
                row, start, end = self.pending.popleft()
                observed = self._wall()
                begin_ms = self.origin.elapsed_time(start)
                end_ms = self.origin.elapsed_time(end)
                upper = observed - end_ms / 1000
                self.wall_upper = min(self.wall_upper or upper, upper)
                row.update(
                    timing_status="complete",
                    gpu_start_ms=begin_ms,
                    gpu_end_ms=end_ms,
                    gpu_seconds=(end_ms - begin_ms) / 1000,
                    gpu_origin_wall_lower=self.wall_lower,
                    gpu_origin_wall_upper=self.wall_upper,
                )
                self._write(row)
        except Exception:
            self._disable()

    def close(self) -> None:
        self.drain()
        if self.closed:
            return
        try:
            for row, _, _ in self.pending:
                self._write({**row, "timing_status": "incomplete_at_shutdown"})
            self.pending.clear()
        except OSError:
            self._disable()
        finally:
            if not self.closed:
                self.closed = True
                self.file.close()


def bind_batch_observer(scheduler: Any) -> BatchObserver | None:
    """Opt in on one TP/CP leader per PP stage; leave scheduling untouched."""
    directory = os.environ.get(TRACE_DIR_ENV)
    if not directory or scheduler.tp_rank != 0 or scheduler.attn_cp_rank != 0:
        return None
    args = scheduler.server_args
    if args.enable_dp_attention or args.speculative_algorithm or args.enable_pdmux:
        raise ValueError(
            "Rollout batch tracing requires ordinary TP/DP, without DP attention, speculative decoding, or PD-Multiplexing"
        )
    observer = BatchObserver(scheduler, directory)
    original = scheduler.record_forward_metrics

    @contextmanager
    def observe(batch):
        item = observer.start(batch)
        succeeded = False
        try:
            with original(batch):
                yield
            succeeded = True
        finally:
            if item is not None:
                item[0]["forward_status"] = "returned" if succeeded else "raised"
            observer.finish(item)

    def drain_after(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            finally:
                observer.drain()

        return wrapped

    scheduler.record_forward_metrics = observe
    # Idle/paused iterations and the last result also drain: no later forward
    # is needed to collect the final completed batch.
    scheduler.recv_requests = drain_after(scheduler.recv_requests)
    scheduler.process_batch_result = drain_after(scheduler.process_batch_result)
    logger.info("Rollout batch trace: %s", observer.path)
    return observer
