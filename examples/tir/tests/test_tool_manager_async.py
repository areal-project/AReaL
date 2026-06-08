# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest
from tool_manager import ToolManager
from tools import ToolCallStatus


def _has_daytona_credentials() -> bool:
    return importlib.util.find_spec("daytona") is not None and bool(
        os.environ.get("DAYTONA_API_KEY")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
@pytest.mark.asyncio
async def test_tool_manager_async_dispatch_daytona():
    manager = ToolManager(enabled_tools="daytona_python")

    try:
        result, status = await manager.aexecute_tool_call("```python\nprint(1)\n```")
    finally:
        await asyncio.to_thread(manager.cleanup)

    assert status == ToolCallStatus.SUCCESS
    assert "1" in result
