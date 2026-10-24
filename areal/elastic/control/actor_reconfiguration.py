# SPDX-License-Identifier: Apache-2.0

"""Explicit Actor DP changes over an unchanged, already allocated GPU pool."""

from __future__ import annotations

import argparse
import fcntl
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from areal.utils.atomic import atomic_json, json_digest

from .config_application import ConfigRequest, ConfigSnapshot


@dataclass(frozen=True)
class ActorDPConfig:
    """Active Actor rank count within the immutable GPU reservation."""

    world_size: int

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("Actor DP size must be a positive integer")


class ActorDPExecutor:
    """Configuration journal adapter; the native runtime owns all GPU operations."""

    def __init__(self, owner: Any):
        self.owner = owner
        self.context_id = json_digest(
            {
                "scope": "native-fsdp-dp-v1",
                "run": owner.identity,
                "queue": owner.inbox.queue.queue_id,
                "actor_gpu_pool": list(owner.options.actor_devices),
                "rollout_gpu_pool": list(owner.options.rollout_devices),
            }
        )

    def observe(self) -> ConfigSnapshot[ActorDPConfig]:
        state = self.owner.control_snapshot()
        return ConfigSnapshot(
            ActorDPConfig(self.owner.actor_world_size),
            state.applied_version,
            state.boundary_id,
            state.boundary_ready,
        )

    def decode_config(self, value: dict) -> ActorDPConfig:
        return ActorDPConfig(**value)

    def validate(self, target: ActorDPConfig) -> None:
        self.owner.validate_actor_dp(target.world_size)

    def apply(self, target: ActorDPConfig) -> None:
        self.owner._switch_actor_dp(target.world_size)

    def recover(self, source: ConfigSnapshot[ActorDPConfig]) -> None:
        if self.observe() != source:
            self.owner._switch_actor_dp(source.config.world_size)
        if self.observe() != source:
            raise RuntimeError(
                "Actor DP recovery did not preserve the source training boundary"
            )


def submit_request(
    state_root: Path,
    *,
    world_size: int,
    request_id: str,
    expected_generation: int,
    expected_weights_version: int,
) -> Path:
    """Queue one local request; runtime services it between completed updates."""
    if not (state_root / "state.json").is_file():
        raise ValueError("State root does not belong to an initialized native runtime")
    request = ConfigRequest(
        request_id,
        ActorDPConfig(world_size),
        expected_generation,
        expected_weights_version,
        "manual",
    )
    root = state_root / "actor-reconfiguration"
    root.mkdir(exist_ok=True)
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "request.json"
        if path.exists():
            raise RuntimeError("An Actor DP request is already pending")
        atomic_json(path, asdict(request))
    return path


def poll_request(owner: Any) -> None:
    """The mailbox is local to the CPU coordinator; no new service or GPU lease."""
    root = owner.root / "actor-reconfiguration"
    if not (root / "request.json").is_file():
        return
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = json.loads((root / "request.json").read_text())
        try:
            if request.pop("policy_bundle_id") != "manual":
                raise ValueError("Actor DP mailbox requires manual provenance")
            target = ActorDPConfig(**request.pop("target"))
            receipt = owner.reconfigure_actor(world_size=target.world_size, **request)
            response = {"request_id": request["request_id"], "receipt": receipt}
        except Exception as error:
            response = {
                "request_id": request.get("request_id"),
                "error": f"{type(error).__name__}: {error}",
            }
            # Validation failures are safe to report. Failed rollback keeps the
            # durable intent and admission fence; do not continue training.
            pending = owner.actor_config_application.state().get("pending")
            atomic_json(root / "response.json", response)
            if pending is not None:
                raise
        atomic_json(root / "response.json", response)
        (root / "request.json").unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    request = sub.add_parser("request")
    request.add_argument("--world-size", type=int, required=True)
    request.add_argument("--request-id", required=True)
    request.add_argument("--expected-generation", type=int, required=True)
    request.add_argument("--expected-weights-version", type=int, required=True)
    args = parser.parse_args()
    root = args.state_root
    if args.command == "status":
        status = {"runtime": json.loads((root / "state.json").read_text())}
        for name in ("state", "request", "response"):
            path = root / "actor-reconfiguration" / f"{name}.json"
            status[name] = json.loads(path.read_text()) if path.exists() else None
        print(json.dumps(status, indent=2))
    else:
        print(
            submit_request(
                root,
                world_size=args.world_size,
                request_id=args.request_id,
                expected_generation=args.expected_generation,
                expected_weights_version=args.expected_weights_version,
            )
        )


if __name__ == "__main__":
    main()
