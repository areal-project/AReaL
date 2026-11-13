# SPDX-License-Identifier: Apache-2.0
"""Bind Hermes 0.19's native tools to an isolated task workspace."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from examples.v2_online_workflows.trace import mark_tool_execution


@contextmanager
def python_thread_pool_compat():
    """Keep Hermes 0.19's daemon workers on CPython 3.14's WorkerContext API."""
    if sys.version_info[:2] != (3, 14):
        yield
        return
    from concurrent.futures.thread import _worker

    from tools.daemon_pool import DaemonThreadPoolExecutor

    original = DaemonThreadPoolExecutor._adjust_thread_count

    def adjust(pool):
        if pool._idle_semaphore.acquire(timeout=0):
            return
        count = len(pool._threads)
        if count < pool._max_workers:

            def gone(_, queue=pool._work_queue):
                queue.put(None)

            worker = threading.Thread(
                name=f"{pool._thread_name_prefix or pool}_{count}",
                target=_worker,
                args=(
                    weakref.ref(pool, gone),
                    pool._create_worker_context(),
                    pool._work_queue,
                ),
                daemon=True,
            )
            worker.start()
            pool._threads.add(worker)

    DaemonThreadPoolExecutor._adjust_thread_count = adjust
    try:
        yield
    finally:
        DaemonThreadPoolExecutor._adjust_thread_count = original


def require_native_tool_execution(result: dict, trace: Any) -> None:
    """Recognize 0.19's synthetic receipts for swallowed outer-loop failures.

    These exact prefixes are emitted by conversation_loop, not shell exits or
    model-authored code errors. Preserve ordinary native tool feedback.
    """
    messages = list(result.get("messages", []))
    messages += [m for node in trace.nodes for m in node.get("tool_messages", [])]
    prefixes = (
        "Error executing tool: Error during OpenAI-compatible API call #",
        "Error executing tool: Error during local message processing after OpenAI-compatible API call #",
    )
    if any(
        isinstance(m.get("content"), str) and m["content"].startswith(prefixes)
        for m in messages
        if m.get("role", "tool") == "tool"
    ):
        raise RuntimeError(
            "Hermes tool dispatcher failed; this is not a zero-reward answer"
        )


@contextmanager
def native_workspace_tools(runtime: Any, trace: Any):
    """One runner process owns one Task; never create another sandbox here.

    Use Hermes' task overrides and execution middleware. The callback observes
    native tools and forwards the original arguments/result without rewriting.
    """
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    from tools.process_registry import process_registry
    from tools.terminal_tool import (
        cleanup_vm,
        clear_task_env_overrides,
        register_task_env_overrides,
        terminal_tool,
    )

    task_id = runtime.task_id
    pool = f"{runtime.kind}:{task_id}"
    # Native file/terminal tools can fan out. Do not claim a serial service pool.
    trace.pools[pool] = None
    manager = get_plugin_manager()
    plugin = PluginContext(PluginManifest(name="areal-task-observation"), manager)
    observing: ContextVar[bool] = ContextVar("hermes_tool_observation", default=False)

    def observe(*, tool_name, args, next_call, task_id="", **context):
        # The concurrent path nests agent and registry execution middleware.
        # Observe the outer execution once, while forwarding both native layers.
        if task_id != runtime.task_id or observing.get():
            return next_call(args)

        def execute():
            started = time.time()
            clock = time.perf_counter()
            mark_tool_execution(started)
            record = {
                "task_id": task_id,
                "name": tool_name,
                "arguments": args,
                "started": started,
                "status": "completed",
            }
            try:
                result = next_call(args)
                record["result"] = result
                if isinstance(result, str):
                    try:
                        value = json.loads(result)
                    except ValueError:
                        value = None
                    if isinstance(value, dict):
                        for key in ("exit_code", "error"):
                            if key in value:
                                record[key] = value[key]
                        if value.get("status") == "infrastructure_error":
                            record["status"] = "infrastructure_error"
                return result
            except Exception as exc:
                record.update(
                    status="infrastructure_error", error_type=type(exc).__name__
                )
                raise
            finally:
                record.update(finished=time.time(), seconds=time.perf_counter() - clock)
                with runtime._lock:
                    record["call"] = len(runtime.records) + 1
                    runtime.records.append(record)
                    runtime.call_timings.append(
                        {
                            key: record[key]
                            for key in ("name", "started", "finished", "seconds")
                        }
                    )
                    with (runtime.directory / "tools.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        token = observing.set(True)
        try:
            return trace.observe_tool(tool_name, pool, execute)
        finally:
            observing.reset(token)

    values = {
        "TERMINAL_ENV": getattr(runtime, "terminal_env", "local"),
        "TERMINAL_CWD": runtime.workdir,
        "PATH": runtime.tool_path,
        "PYTHONDONTWRITEBYTECODE": "1",
        # Keep the packaged Hermes dependencies out of the project's Python.
        "PYTHONPATH": runtime.python_path,
    }
    saved = {key: os.environ.get(key) for key in values}
    registered = False
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        register_task_env_overrides(
            task_id,
            getattr(
                runtime,
                "terminal_overrides",
                {"env_type": "local", "cwd": runtime.workdir},
            ),
        )
        # Login startup files can replace PATH in Hermes' initial shell snapshot.
        # Seed the selected test environment through a native terminal command.
        setup = "export PATH=" + shlex.quote(runtime.tool_path)
        setup += (
            "; export PYTHONPATH=" + shlex.quote(runtime.python_path)
            if runtime.python_path is not None
            else "; unset PYTHONPATH"
        )
        result = json.loads(terminal_tool(setup, task_id=task_id, force=True))
        if result.get("exit_code") != 0 or result.get("error"):
            (runtime.directory / "initialization-error.json").write_text(
                json.dumps(result)
            )
            raise RuntimeError("Cannot initialize native Hermes workspace tools")
        plugin.register_middleware("tool_execution", observe)
        registered = True
        with python_thread_pool_compat():
            yield
    finally:
        if registered:
            # 0.19 exposes registration but no unregister API. Remove only ours.
            manager._middleware["tool_execution"].remove(observe)
        try:
            process_registry.kill_all(task_id=task_id)
            cleanup_vm(task_id)
        finally:
            clear_task_env_overrides(task_id)
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


# Keep the existing import available for previously authored callers.
native_repository_tools = native_workspace_tools
