# SPDX-License-Identifier: Apache-2.0

"""Sandbox backends for code execution in rollouts and user scripts.

Currently provides Daytona and OpenSandbox as cloud sandboxes. Install via::

    uv sync --extra sandbox
"""

from ._client import DaytonaClientManager
from ._opensandbox_session import (
    DEFAULT_TTL_SECONDS,
    OpenSandboxCellError,
    OpenSandboxCellResult,
    OpenSandboxExecResult,
    OpenSandboxSession,
    load_opensandbox_sdk,
)
from .opensandbox_runner import OpenSandboxRunner, OpenSandboxRunResult
from .runner import DaytonaRunner, DaytonaRunResult

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DaytonaClientManager",
    "DaytonaRunResult",
    "DaytonaRunner",
    "OpenSandboxCellError",
    "OpenSandboxCellResult",
    "OpenSandboxExecResult",
    "OpenSandboxRunResult",
    "OpenSandboxRunner",
    "OpenSandboxSession",
    "load_opensandbox_sdk",
]
