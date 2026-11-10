# SPDX-License-Identifier: Apache-2.0
"""Opt-in host-side Apptainer tools. The agent image only needs the socket client.

The compute host runs ``singularity_host serve`` with Python 3.6.8+.
No model, inference key, or training dependency is needed by the host process.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict
from typing import Any

from examples.v2_online_workflows.environments.singularity_host import Host as Host
from examples.v2_online_workflows.environments.singularity_host import (
    HostServer as HostServer,
)
from examples.v2_online_workflows.environments.singularity_host import (
    main as serve_main,
)
from examples.v2_online_workflows.trace import mark_tool_execution

_MAX_MESSAGE = 128 * 1024 * 1024


def rpc(action: str, **payload: Any) -> Any:
    """A private same-user Unix socket, mounted only into the agent/client image."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(
            max(
                float(os.environ.get("AREAL_TASK_HOST_TIMEOUT", "900")),
                float(payload.get("timeout", 0)) + 120,
            )
        )
        connection.connect(os.environ["AREAL_TASK_HOST_SOCKET"])
        connection.sendall(json.dumps({"action": action, **payload}).encode() + b"\n")
        with connection.makefile("rb") as stream:
            response = _read(stream)
    if "error" in response:
        raise RuntimeError("Singularity host: " + response["error"])
    return response["result"]


def _read(stream: Any) -> dict[str, Any]:
    row = stream.readline(_MAX_MESSAGE + 1)
    if not row.endswith(b"\n") or len(row) > _MAX_MESSAGE:
        raise ValueError("Incomplete or oversized host message")
    return json.loads(row)


def create_environment(
    provider: Any, settings: Any, task_id: str, *, image_id: str, **options: Any
) -> Any:
    backend = os.environ.get("AREAL_TASK_RUNTIME", "docker")
    if backend == "docker":
        return provider(settings, task_id, image_id=image_id, **options)
    if backend != "singularity":
        raise ValueError(f"Unknown task runtime: {backend}")
    return RemoteEnvironment(provider, settings, task_id, image_id, options)


class RemoteEnvironment:
    """The existing harness contract; no container executable runs in this process."""

    def __init__(
        self,
        provider: Any,
        settings: Any,
        task_id: str,
        image_id: str,
        options: dict[str, Any],
    ) -> None:
        self.kind, self.toolset = provider.kind, provider.toolset
        self.tool_schemas = provider.tool_schemas
        self.settings, self.task_id = settings, task_id
        self.directory = settings.root / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._closed, self._metadata = False, {}
        rpc(
            "open",
            kind=self.kind,
            task_id=task_id,
            image=image_id,
            settings={**asdict(settings), "root": str(settings.root)},
            options=options,
        )

    def execute(self, value: str) -> str:
        return self.invoke(
            "math_python" if self.kind == "python" else "workspace_exec",
            {"code" if self.kind == "python" else "command": value},
        )

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        result = rpc("invoke", task_id=self.task_id, name=name, arguments=arguments)
        timing = result["timing"]
        if timing:
            mark_tool_execution(
                timing["started"], seconds=timing["finished"] - timing["started"]
            )
        return result["value"]

    def export_artifacts(self) -> dict[str, Any]:
        return rpc("export", task_id=self.task_id)

    def metadata(self) -> dict[str, Any]:
        return self._metadata if self._closed else rpc("metadata", task_id=self.task_id)

    def close(self) -> None:
        if not self._closed:
            self._metadata = rpc("close", task_id=self.task_id)
            self._closed = True


def main() -> None:
    # Keep the original modern-Python CLI entry point available.
    import sys

    if sys.argv[1:] == ["health"]:
        print(json.dumps(rpc("health")))
    else:
        serve_main()


if __name__ == "__main__":
    main()
