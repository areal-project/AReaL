# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import areal.infra.sandbox._client as client_module
from areal.infra.sandbox._client import DaytonaClientManager


class FakeDaytonaConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAsyncDaytona:
    def __init__(self, config):
        self.config = config
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def reset_daytona_client_manager(monkeypatch):
    DaytonaClientManager._client = None
    DaytonaClientManager._config_overrides = {}
    DaytonaClientManager._atexit_registered = False
    DaytonaClientManager._loop = None
    monkeypatch.setattr(client_module, "AsyncDaytona", FakeAsyncDaytona, raising=False)
    monkeypatch.setattr(
        client_module, "DaytonaConfig", FakeDaytonaConfig, raising=False
    )
    yield
    DaytonaClientManager._client = None
    DaytonaClientManager._config_overrides = {}
    DaytonaClientManager._atexit_registered = False
    DaytonaClientManager._loop = None


@pytest.mark.asyncio
async def test_get_client_returns_singleton():
    first_client = await DaytonaClientManager.get_client()
    second_client = await DaytonaClientManager.get_client()

    assert first_client is second_client
    assert isinstance(first_client, FakeAsyncDaytona)
    assert first_client.config.kwargs == {"connection_pool_maxsize": None}


@pytest.mark.asyncio
async def test_configure_before_first_use_applies_overrides():
    DaytonaClientManager.configure(api_url="https://example.daytona.test")

    client = await DaytonaClientManager.get_client()

    assert client.config.kwargs == {
        "connection_pool_maxsize": None,
        "api_url": "https://example.daytona.test",
    }


@pytest.mark.asyncio
async def test_configure_after_init_raises():
    await DaytonaClientManager.get_client()

    with pytest.raises(RuntimeError, match="already initialized"):
        DaytonaClientManager.configure(api_url="https://example.daytona.test")


@pytest.mark.asyncio
async def test_atexit_registered_on_first_get_client(monkeypatch):
    registrations: list[object] = []
    monkeypatch.setattr(client_module.atexit, "register", registrations.append)

    await DaytonaClientManager.get_client()
    await DaytonaClientManager.get_client()

    assert len(registrations) == 1


@pytest.mark.asyncio
async def test_close_is_idempotent():
    client = await DaytonaClientManager.get_client()

    await DaytonaClientManager.close()
    await DaytonaClientManager.close()

    assert client.close_calls == 1
    assert DaytonaClientManager._client is None
