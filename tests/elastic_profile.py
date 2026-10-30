# SPDX-License-Identifier: Apache-2.0
"""Test-only span hooks. No config/arguments, credentials or exception text recorded."""

import functools
import inspect
import json
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


class Recorder:
    def __init__(self, root, component, generation=0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.component, self.generation = component, generation

    @contextmanager
    def span(self, name):
        start = time.monotonic_ns()
        status = "ok"
        try:
            yield
        except BaseException as error:
            status = type(error).__name__
            raise
        finally:
            end = time.monotonic_ns()
            row = dict(
                name=name,
                component=self.component,
                generation=self.generation,
                start_ns=start,
                end_ns=end,
                seconds=(end - start) / 1e9,
                pid=os.getpid(),
                thread=threading.get_ident(),
                status=status,
                span_id=uuid.uuid4().hex,
            )
            # Separate process files also capture native writer threads/children.
            fd = os.open(
                self.root / f"{self.component}-{os.getpid()}.jsonl",
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
            try:
                os.write(fd, (json.dumps(row) + "\n").encode())
            finally:
                os.close(fd)

    def wrap(self, function, name):
        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def wrapped(*args, **kwargs):
                label = name(*args, **kwargs) if callable(name) else name
                with self.span(label):
                    return await function(*args, **kwargs)
        else:

            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                label = name(*args, **kwargs) if callable(name) else name
                with self.span(label):
                    return function(*args, **kwargs)

        return wrapped


def hook(stack, target, name, recorder, label):
    function = getattr(target, name)
    replacement = recorder.wrap(function, label)
    if isinstance(target, type) and isinstance(
        inspect.getattr_static(target, name), staticmethod
    ):
        replacement = staticmethod(replacement)
    stack.enter_context(patch.object(target, name, replacement))


class OSProxy:
    """Replace one module binding, never the shared os module's functions."""

    def __init__(self, delegate, recorder, prefix):
        self.delegate, self.recorder, self.prefix = delegate, recorder, prefix

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def fsync(self, fd):
        kind = "directory" if stat.S_ISDIR(self.delegate.fstat(fd).st_mode) else "file"
        with self.recorder.span(f"{self.prefix}.fsync_{kind}"):
            return self.delegate.fsync(fd)

    def replace(self, *args, **kwargs):
        with self.recorder.span(f"{self.prefix}.rename"):
            return self.delegate.replace(*args, **kwargs)


def instrument_controller(stack, actor, recorder):
    for name in (
        "create_process_group",
        "initialize",
        "full_checkpoint",
        "connect_engine",
        "destroy",
    ):
        hook(stack, actor, name, recorder, f"controller.{name}")
    for name in ("create_workers", "get_workers", "delete_workers"):

        def label(*args, _name=name, **kwargs):
            role = getattr(kwargs.get("job"), "role", kwargs.get("role", "unknown"))
            return f"scheduler.{_name}.{role}"

        hook(stack, actor.scheduler, name, recorder, label)
    hook(stack, actor, "_async_set_guards_env", recorder, "controller.rendezvous_env")
    hook(
        stack,
        actor,
        "_async_fork_on_guard",
        recorder,
        lambda *a, **kw: f"controller.fork.{kw['role']}.{kw['worker_index']}",
    )
    hook(stack, actor, "_create_engine_on_worker", recorder, "controller.create_engine")
    hook(
        stack,
        actor,
        "_call_worker_engine_endpoint",
        recorder,
        lambda addr, endpoint, **kw: "rpc" + endpoint.replace("/", "."),
    )
    hook(stack, actor, "_register_in_router", recorder, "controller.register")


def instrument_store(stack, recorder):
    from areal.elastic.state import checkpoint_store as store

    stack.enter_context(patch.object(store, "os", OSProxy(store.os, recorder, "store")))
    hook(
        stack,
        store,
        "atomic_json",
        recorder,
        lambda path, value: "store.publish." + Path(path).name,
    )


def union_seconds(intervals):
    """Union, not sum: ranks/threads can overlap and nested spans double count."""
    total, end = 0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total / 1e9


def spans_in_window(rows, window):
    """Exclude neighboring audit calls; clip intersecting spans to measured work."""
    selected = []
    for row in rows:
        start = max(row["start_ns"], window["start_ns"])
        end = min(row["end_ns"], window["end_ns"])
        if end > start:
            selected.append(
                {
                    **row,
                    "start_ns": start,
                    "end_ns": end,
                    "seconds": (end - start) / 1e9,
                }
            )
    return selected


def summarize_spans(root):
    rows = [
        json.loads(line)
        for p in Path(root).glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    summary = {}
    for generation, name in sorted({(row["generation"], row["name"]) for row in rows}):
        selected = [
            r for r in rows if r["generation"] == generation and r["name"] == name
        ]
        summary[f"g{generation}/{name}"] = dict(
            calls=len(selected),
            max_seconds=max(r["seconds"] for r in selected),
            union_seconds=union_seconds(
                [(r["start_ns"], r["end_ns"]) for r in selected]
            ),
            envelope_seconds=(
                max(r["end_ns"] for r in selected)
                - min(r["start_ns"] for r in selected)
            )
            / 1e9,
            errors=[r["status"] for r in selected if r["status"] != "ok"],
        )
    return summary


def phase_breakdown(root):
    """Measured-window decomposition; audit spans are deliberately excluded."""
    rows = [
        json.loads(line)
        for p in Path(root).glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    output = {
        "restore_cycles": [],
        "save": None,
        "accounting": "Single-host wall time. Parallel rank intervals use unions; nested spans are not added twice.",
    }
    prefixes = {
        "coordinator_gloo": ("controller.create_process_group",),
        "guards": (
            "scheduler.create_workers.actor-guard",
            "scheduler.get_workers.actor-guard",
            "controller.rendezvous_env",
        ),
        "worker_services": ("controller.fork.train-worker",),
        "engine_creation": ("controller.create_engine",),
        "distributed_groups": ("rpc.create_process_group",),
        "model_optimizer_initialization": ("rpc.initialize",),
        "router": ("controller.fork.router",),
        "data_proxy": ("controller.fork.data-proxy",),
        "gateway": ("controller.fork.gateway",),
        "checkpoint_load_and_preflight": ("controller.full_checkpoint",),
        "awex_connection": ("controller.connect_engine",),
    }
    for window in sorted(rows, key=lambda r: r["start_ns"]):
        if window["name"] not in ("bench.restore_complete", "bench.save_complete"):
            continue
        if window["status"] != "ok":
            continue
        selected = spans_in_window(rows, window)
        if window["name"] == "bench.restore_complete":
            selected = [r for r in selected if r["component"] == "coordinator"]
            parts, intervals = {}, []
            for label, names in prefixes.items():
                spans = [r for r in selected if r["name"].startswith(names)]
                ranges = [(r["start_ns"], r["end_ns"]) for r in spans]
                parts[label] = union_seconds(ranges)
                intervals.extend(ranges)
            assert abs(sum(parts.values()) - union_seconds(intervals)) < 0.01
            parts["other"] = window["seconds"] - union_seconds(intervals)
            output["restore_cycles"].append(
                dict(
                    generation=window["generation"],
                    total_seconds=window["seconds"],
                    parts=parts,
                )
            )
        else:
            critical = max(
                (r for r in selected if r["name"] == "engine.save"),
                key=lambda r: r["end_ns"],
            )
            rank_rows = [r for r in selected if r["component"] == critical["component"]]

            def duration(name):
                return union_seconds(
                    [
                        (r["start_ns"], r["end_ns"])
                        for r in rank_rows
                        if r["name"] == name
                    ]
                )

            # Nested writer time is split explicitly, on the last finishing rank.
            bucket = duration("save.bucket_write_and_sync")
            fsync = duration("writer.fsync_file")
            writer = duration("save.write_and_sync")
            parts = dict(
                state_dict=duration("checkpointer.generate_state_dict"),
                planning=duration("save.plan_and_prepare"),
                device_to_host=duration("save.device_to_host"),
                serialization_and_file_write=bucket - fsync,
                writer_fsync=fsync,
                writer_dispatch_and_join=writer - bucket,
                metadata_finish=duration("save.metadata_finish"),
            )
            parts["other_including_rpc_and_outer_commit"] = window["seconds"] - sum(
                parts.values()
            )
            output["save"] = dict(
                total_seconds=window["seconds"],
                critical_rank=critical["component"],
                parts=parts,
                outer_file_directory_fsync_seconds=sum(
                    r["seconds"]
                    for r in selected
                    if r["name"].startswith("store.fsync_")
                ),
            )
    return output
