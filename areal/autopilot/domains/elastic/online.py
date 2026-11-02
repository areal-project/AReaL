# SPDX-License-Identifier: Apache-2.0

"""Run bounded Online Reference revisions from explicitly supplied committed work."""

from __future__ import annotations

import fcntl
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from areal.elastic.infra.activation import (
    ReconfiguratorBoundaryBusy,
    ReconfiguratorRequest,
)
from areal.utils.atomic import atomic_json

from .cycle import ReconfiguratorRevisionCycle, _fixed_json


class RuntimeRevisionLoop:
    """One background CPU cycle; the caller maps existing measurements to problems.

    problem_builder(snapshot, assembly, directory) returns (model_evaluator,
    optimizer_cases). It must preserve actual configuration identities and the
    calibration/validation split. Missing serving measurements are not synthesized.
    """

    def __init__(
        self,
        root,
        *,
        problem_builder,
        model_agent,
        optimizer_agent,
        every_updates=4,
        max_windows=3,
        max_model_rounds=2,
        max_optimizer_rounds=2,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (
                every_updates,
                max_windows,
                max_model_rounds,
                max_optimizer_rounds,
            )
        ):
            raise ValueError("Runtime revision budgets must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.builder = problem_builder
        self.agents = model_agent, optimizer_agent
        self.every, self.maximum = every_updates, max_windows
        self.rounds = max_model_rounds, max_optimizer_rounds
        self._lock = (self.root / "runtime.lock").open("a")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _fixed_json(
            self.root / "context.json",
            dict(
                every_updates=every_updates,
                max_windows=max_windows,
                max_model_rounds=max_model_rounds,
                max_optimizer_rounds=max_optimizer_rounds,
            ),
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="online-revision"
        )
        self.future = self.pending = None

    def _run(self, directory, request):
        model, cases = self.builder(request["snapshot"], request["assembly"], directory)
        model_source = optimizer_source = None
        if request["parent_path"] is not None:
            from .bundle import validate_bundle

            parent = json.loads(Path(request["parent_path"]).read_text())
            validate_bundle(parent)
            model_source = parent["components"]["model"]["source"]
            optimizer_source = parent["components"]["optimizer"]["source"]
        result = ReconfiguratorRevisionCycle(
            model,
            cases,
            directory / "cycle",
            assembly=request["assembly"],
            initial_model_source=model_source,
            initial_optimizer_source=optimizer_source,
            parent_bundle=request["parent_id"]
            if request["parent_path"] is not None
            else None,
        ).run(
            *self.agents,
            max_model_rounds=self.rounds[0],
            max_optimizer_rounds=self.rounds[1],
        )
        return directory, request, result

    def tick(self, host):
        if self._lock.closed:
            raise RuntimeError("Runtime revision loop is closed")
        if not host.measurement_ready:
            return
        application = host.reconfigurator_application
        try:
            _fixed_json(
                self.root / "execution.json",
                dict(
                    application=application.context,
                    history=host.runtime_history.context,
                ),
            )
        except (OSError, ValueError, TypeError) as error:
            host.measurement_failed(error)
            return
        if self.future is not None and self.future.done():
            try:
                self.pending = self.future.result()
            except Exception as error:
                atomic_json(
                    self.running / "outcome.json",
                    dict(status="failed", error=f"{type(error).__name__}: {error}"),
                )
            self.future = None
        if self.pending is not None:
            directory, request, result = self.pending
            if result["bundle"] is None:
                atomic_json(directory / "outcome.json", dict(status="unchanged"))
                self.pending = None
            elif host.reconfigurator_boundary_ready:
                state = application.state()
                old = state["receipts"].get(directory.name)
                if old is not None:
                    atomic_json(
                        directory / "outcome.json", dict(status="applied", receipt=old)
                    )
                    self.pending = None
                elif state["generation"] != request["generation"]:
                    atomic_json(directory / "outcome.json", dict(status="superseded"))
                    self.pending = None
                else:
                    prepared = host.prepare_reconfigurator(Path(result["bundle"]))
                    try:
                        receipt = host.apply_reconfigurator(
                            ReconfiguratorRequest(
                                directory.name,
                                prepared.bundle_id,
                                state["generation"],
                                host.weights_version,
                            ),
                            prepared,
                        )
                    except ReconfiguratorBoundaryBusy:
                        return
                    atomic_json(
                        directory / "outcome.json",
                        dict(status="applied", receipt=receipt),
                    )
                    self.pending = None
        if self.future is not None or self.pending is not None:
            return
        windows = sorted(self.root.glob("window-*"))
        unfinished = [p for p in windows if not (p / "outcome.json").exists()]
        if unfinished:
            directory = unfinished[0]
            request = json.loads((directory / "request.json").read_text())
        else:
            if len(windows) >= self.maximum:
                return
            version = host.weights_version
            last = max(
                (
                    json.loads((p / "request.json").read_text())["version"]
                    for p in windows
                ),
                default=0,
            )
            if version - last < self.every:
                return
            state = application.state()
            try:
                snapshot = host.runtime_history.snapshot(version)
                assembly = host.assembly()
            except (OSError, ValueError, TypeError) as error:
                host.measurement_failed(error)
                return
            if not snapshot["updates"]:
                return
            directory = self.root / f"window-{len(windows) + 1:04d}"
            directory.mkdir()
            parent = application.root / "bundles" / state["bundle_id"] / "bundle.json"
            request = dict(
                version=version,
                generation=state["generation"],
                snapshot=snapshot,
                assembly=assembly,
                parent_id=state["bundle_id"],
                parent_path=str(parent) if parent.exists() else None,
            )
            atomic_json(directory / "request.json", request)
        self.running = directory
        self.future = self.executor.submit(self._run, directory, request)

    def close(self):
        if self._lock.closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=False)
        self._lock.close()
