# SPDX-License-Identifier: Apache-2.0

"""Sandbox backends for code execution in rollouts and user scripts.

Currently provides Daytona and OpenSandbox as cloud sandboxes. Install via::

    uv sync --extra sandbox
"""

from ._client import DaytonaClientManager
from .opensandbox_runner import OpenSandboxRunner, OpenSandboxRunResult
from .runner import DaytonaRunner, DaytonaRunResult

__all__ = [
    "DaytonaClientManager",
    "DaytonaRunResult",
    "DaytonaRunner",
    "OpenSandboxRunResult",
    "OpenSandboxRunner",
]
