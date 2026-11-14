# SPDX-License-Identifier: Apache-2.0

"""Sandbox backends for code execution in rollouts and user scripts.

Provides Daytona and Runta as cloud sandboxes. Install via::

    uv sync --extra sandbox                    # Daytona
    uv pip install "runta-sdk>=0.2.6,<0.3"     # Runta
"""

from ._client import DaytonaClientManager
from .runner import DaytonaRunner, DaytonaRunResult
from .runta import AsyncRuntaRunner, RuntaRunner, RuntaRunResult

__all__ = [
    "AsyncRuntaRunner",
    "DaytonaClientManager",
    "DaytonaRunResult",
    "DaytonaRunner",
    "RuntaRunResult",
    "RuntaRunner",
]
