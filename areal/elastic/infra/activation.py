# SPDX-License-Identifier: Apache-2.0

"""Durable reconfigurator activation with an injected bundle loader and verifier."""

from __future__ import annotations

import fcntl
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.elastic.contracts import BatchWorkload, HistoryWindow, Measurement
from areal.utils.atomic import atomic_json
from areal.utils.atomic import json_digest as digest


@dataclass(frozen=True)
class ReconfiguratorRequest:
    request_id: str
    bundle_id: str
    expected_generation: int
    expected_weights_version: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.request_id
        ):
            raise ValueError("Invalid reconfigurator request identity")
        if not isinstance(self.bundle_id, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.bundle_id
        ):
            raise ValueError("Reconfigurator bundle identity must be a SHA256")
        if any(
            type(n) is not int or n < 0
            for n in (self.expected_generation, self.expected_weights_version)
        ):
            raise ValueError("Expected generations/versions must be nonnegative")


class ReconfiguratorBoundaryBusy(RuntimeError):
    """A valid reconfigurator application must wait for the next planning boundary."""


class ReconfiguratorApplication:
    """A loop owns admission; this store owns the immutable active-reconfigurator pointer."""

    def __init__(
        self,
        root: Path,
        *,
        compatibility: dict,
        execution_context: dict,
        initial_bundle_id: str = "constructor-reconfigurator",
        prepare_bundle,
        validate_bundle,
    ):
        self._prepare_bundle, self._validate_bundle = prepare_bundle, validate_bundle
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.compatibility = json.loads(json.dumps(compatibility, allow_nan=False))
        self.execution_context = json.loads(
            json.dumps(execution_context, allow_nan=False)
        )
        if not initial_bundle_id or not all(
            self.execution_context.get(k) for k in ("queue_id", "checkpoint_root")
        ):
            raise ValueError(
                "Reconfigurator activation requires queue, checkpoint and compatible execution identities"
            )
        self.initial_bundle_id = initial_bundle_id
        self.context = digest(
            {
                "compatibility": self.compatibility,
                "initial": initial_bundle_id,
                "execution": self.execution_context,
            }
        )
        with self._lock():
            if not (self.root / "active.json").exists():
                atomic_json(
                    self.root / "active.json",
                    {
                        "format_version": 1,
                        "context": self.context,
                        "generation": 0,
                        "bundle_id": initial_bundle_id,
                        "receipts": {},
                    },
                )
            self.state()

    def _lock(self):
        lock = (self.root / "reconfigurator.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock.close()
            raise
        return lock

    def state(self):
        value = json.loads((self.root / "active.json").read_text())
        if value["format_version"] != 1 or value["context"] != self.context:
            raise ValueError(
                "Reconfigurator store belongs to another execution context"
            )
        return value

    def preflight(self, request: ReconfiguratorRequest, version: int):
        state = self.state()
        receipt = state["receipts"].get(request.request_id)
        if receipt is not None:
            if receipt["request"] != asdict(request):
                raise ValueError("Reconfigurator request ID was reused")
            return receipt
        if (
            request.expected_generation != state["generation"]
            or request.expected_weights_version != version
        ):
            raise ValueError("Stale reconfigurator generation or weights version")
        return None

    def prepare(self, bundle: Path):
        return self._prepare_bundle(bundle)

    def commit(
        self,
        request: ReconfiguratorRequest,
        version: int,
        prepared,
        history,
        *,
        joint_state=None,
    ):
        with self._lock():
            receipt = self.preflight(request, version)
            if receipt is not None:
                return receipt
            state = self.state()
            if (
                prepared.context_id != self.context
                or prepared.bundle_id != request.bundle_id
                or not prepared.probed
            ):
                raise ValueError(
                    "Prepared reconfigurator belongs to another request/context"
                )
            bundle = self.root / "bundles" / request.bundle_id / "bundle.json"
            value = json.loads(bundle.read_text())
            self._validate_bundle(value)
            if value["bundle_id"] != request.bundle_id or value != json.loads(
                prepared.manifest_json
            ):
                raise ValueError("Prepared reconfigurator bundle changed")
            if (
                value.get("parent_bundle_id") is not None
                and value["parent_bundle_id"] != state["bundle_id"]
            ):
                raise ValueError(
                    "Reconfigurator bundle was revised from another active parent"
                )
            receipt = {
                "request": asdict(request),
                "parent_bundle_id": state["bundle_id"],
                "generation": state["generation"] + 1,
                "weights_version": version,
                "status": "applied",
            }
            atomic_json(
                self.root / "active.json",
                {
                    **state,
                    "generation": receipt["generation"],
                    "bundle_id": request.bundle_id,
                    "receipts": {**state["receipts"], request.request_id: receipt},
                    "history": asdict(history.snapshot()),
                    "joint_state": joint_state,
                },
            )
            return receipt

    def save_history(self, history: HistoryWindow, version: int, *, joint_state=None):
        with self._lock():
            state = self.state()
            atomic_json(
                self.root / "history.json",
                {
                    "context": self.context,
                    "generation": state["generation"],
                    "bundle_id": state["bundle_id"],
                    "weights_version": version,
                    "history": asdict(history.snapshot()),
                    "joint_state": joint_state,
                },
            )

    def history(self, version: int):
        state = self.state()
        if any(
            receipt["weights_version"] > version
            for receipt in state["receipts"].values()
        ):
            raise ValueError(
                "Reconfigurator activation is newer than the recovered checkpoint"
            )
        snapshot = state.get("history", {})
        path = self.root / "history.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["context"] != self.context or saved["weights_version"] > version:
                raise ValueError(
                    "Runtime history belongs to another committed execution"
                )
            if (
                saved["generation"] == state["generation"]
                and saved["bundle_id"] == state["bundle_id"]
            ):
                snapshot = saved["history"]
        result = HistoryWindow(
            max(
                32,
                len(snapshot.get("workloads", ())),
                (len(snapshot.get("measurements", ())) + 15) // 16,
            )
        )
        for row in snapshot.get("workloads", ()):
            result.add_workload(BatchWorkload(**row))
        for row in snapshot.get("measurements", ()):
            result.add_measurement(Measurement(**row))
        return result

    def joint_state(self, version):
        state = self.state()
        selected = state.get("joint_state")
        path = self.root / "history.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["context"] != self.context or saved["weights_version"] > version:
                raise ValueError(
                    "Joint runtime state belongs to another execution/version"
                )
            if (
                saved["generation"] == state["generation"]
                and saved["bundle_id"] == state["bundle_id"]
            ):
                selected = saved.get("joint_state", selected)
        if selected is not None and selected["through_version"] > version:
            raise ValueError("Joint runtime state is newer than recovered weights")
        return selected

    def recover(self):
        with self._lock():
            state = self.state()
            if state["generation"] == 0:
                return None
            return self.prepare(
                self.root / "bundles" / state["bundle_id"] / "bundle.json"
            )


@dataclass(frozen=True)
class PreparedReconfigurator:
    manifest_json: str
    context_id: str
    reconfigurator: object
    probed: bool = False

    @property
    def bundle_id(self):
        return json.loads(self.manifest_json)["bundle_id"]
