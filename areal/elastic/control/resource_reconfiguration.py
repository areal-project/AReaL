# SPDX-License-Identifier: Apache-2.0

"""Explicit local Actor/Rollout leases within an immutable job reservation."""

from __future__ import annotations

import argparse
import fcntl
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.utils.atomic import atomic_json, json_digest

from .config_application import ConfigRequest, ConfigSnapshot


@dataclass(frozen=True)
class ResourceLayout:
    world_size: int
    actor_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]

    def __post_init__(self):
        devices = self.actor_devices + self.rollout_devices
        if (
            type(self.world_size) is not int
            or not 1 <= self.world_size <= len(self.actor_devices)
            or not self.rollout_devices
            or len(set(devices)) != len(devices)
            or any(type(device) is not int or device < 0 for device in devices)
        ):
            raise ValueError(
                "Require disjoint nonempty GPU leases and an Actor DP count fitting its lease"
            )

    @classmethod
    def decode(cls, value):
        return cls(
            value["world_size"],
            tuple(value["actor_devices"]),
            tuple(value["rollout_devices"]),
        )


class ResourceExecutor:
    def __init__(self, owner):
        self.owner = owner
        self.context_id = json_digest(
            {
                "scope": "native-resources-v1",
                "run": owner.identity,
                "queue": owner.inbox.queue.queue_id,
                "reservation": list(owner.resource_reservation),
            }
        )

    def observe(self):
        state = self.owner.control_snapshot()
        return ConfigSnapshot(
            self.owner.resource_layout,
            state.applied_version,
            state.boundary_id,
            state.boundary_ready,
        )

    def decode_config(self, value):
        return ResourceLayout.decode(value)

    def validate(self, target):
        self.owner.validate_resource_layout(target)

    def apply(self, target):
        self.owner._switch_resources(target)

    def recover(self, source):
        if self.observe() != source:
            self.owner._switch_resources(source.config)
        if self.observe() != source:
            raise RuntimeError(
                "Resource recovery did not preserve the source training boundary"
            )


def submit_request(
    state_root: Path,
    *,
    target: ResourceLayout,
    request_id: str,
    expected_generation: int,
    expected_weights_version: int,
) -> Path:
    if not (state_root / "state.json").is_file():
        raise ValueError("State root does not belong to an initialized native runtime")
    request = ConfigRequest(
        request_id, target, expected_generation, expected_weights_version, "manual"
    )
    root = state_root / "resource-reconfiguration"
    root.mkdir(exist_ok=True)
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "request.json"
        if path.exists():
            raise RuntimeError("A resource request is already pending")
        atomic_json(path, asdict(request))
    return path


def poll_request(owner) -> None:
    root = owner.root / "resource-reconfiguration"
    if not (root / "request.json").is_file():
        return
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = json.loads((root / "request.json").read_text())
        try:
            if request.pop("policy_bundle_id") != "manual":
                raise ValueError("Resource mailbox requires manual provenance")
            target = ResourceLayout.decode(request.pop("target"))
            response = {
                "request_id": request["request_id"],
                "receipt": owner.reconfigure_resources(target=target, **request),
            }
        except Exception as error:
            response = {
                "request_id": request.get("request_id"),
                "error": f"{type(error).__name__}: {error}",
            }
            atomic_json(root / "response.json", response)
            application = owner.resource_config_application
            if (
                application is not None
                and application.state().get("pending") is not None
            ):
                raise
        atomic_json(root / "response.json", response)
        (root / "request.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    request = sub.add_parser("request")
    request.add_argument("--actor-devices", type=int, nargs="+", required=True)
    request.add_argument("--rollout-devices", type=int, nargs="+", required=True)
    request.add_argument("--world-size", type=int, required=True)
    request.add_argument("--request-id", required=True)
    request.add_argument("--expected-generation", type=int, required=True)
    request.add_argument("--expected-weights-version", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        status = {
            name: json.loads(path.read_text()) if path.exists() else None
            for name in ("state", "request", "response")
            for path in [args.state_root / "resource-reconfiguration" / f"{name}.json"]
        }
        status["authority"] = "resource" if status["state"] is not None else "actor"
        if status["state"] is None:
            legacy = args.state_root / "actor-reconfiguration/state.json"
            status["state"] = (
                json.loads(legacy.read_text()) if legacy.exists() else {"generation": 0}
            )
        print(json.dumps(status, indent=2))
    else:
        print(
            submit_request(
                args.state_root,
                target=ResourceLayout(
                    args.world_size,
                    tuple(args.actor_devices),
                    tuple(args.rollout_devices),
                ),
                request_id=args.request_id,
                expected_generation=args.expected_generation,
                expected_weights_version=args.expected_weights_version,
            )
        )


if __name__ == "__main__":
    main()
