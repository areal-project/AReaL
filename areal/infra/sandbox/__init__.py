# SPDX-License-Identifier: Apache-2.0

"""Sandbox backends for code execution in rollouts and user scripts.

Provides Daytona and boxd as cloud sandboxes. Install via::

    uv sync --extra sandbox
"""

from ._boxd_client import BoxdClientManager
from ._client import DaytonaClientManager
from .boxd_runner import BoxdRunner, BoxdRunResult
from .runner import DaytonaRunner, DaytonaRunResult

__all__ = [
    "BoxdClientManager",
    "BoxdRunResult",
    "BoxdRunner",
    "DaytonaClientManager",
    "DaytonaRunResult",
    "DaytonaRunner",
]
