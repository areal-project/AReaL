# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import areal.infra.sandbox._boxd_client as client_module
from areal.infra.sandbox._boxd_client import BoxdClientManager


class FakeAsyncBoxd:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def reset_boxd_client_manager(monkeypatch):
    BoxdClientManager._client = None
    BoxdClientManager._config_overrides = {}
    BoxdClientManager._atexit_registered = False
    BoxdClientManager._loop = None
    monkeypatch.setattr(client_module, "AsyncBoxd", FakeAsyncBoxd, raising=False)
    yield
    BoxdClientManager._client = None
    BoxdClientManager._config_overrides = {}
    BoxdClientManager._atexit_registered = False
    BoxdClientManager._loop = None


@pytest.mark.asyncio
async def test_get_client_returns_singleton():
    first_client = await BoxdClientManager.get_client()
    second_client = await BoxdClientManager.get_client()

    assert first_client is second_client
    assert isinstance(first_client, FakeAsyncBoxd)
    assert first_client.kwargs == {}


@pytest.mark.asyncio
async def test_configure_before_first_use_applies_overrides():
    BoxdClientManager.configure(region="fra", timeout=30.0)

    client = await BoxdClientManager.get_client()

    assert client.kwargs == {"region": "fra", "timeout": 30.0}


@pytest.mark.asyncio
async def test_configure_after_init_raises():
    await BoxdClientManager.get_client()

    with pytest.raises(RuntimeError, match="already initialized"):
        BoxdClientManager.configure(region="fra")


@pytest.mark.asyncio
async def test_atexit_registered_on_first_get_client(monkeypatch):
    registrations: list[object] = []
    monkeypatch.setattr(client_module.atexit, "register", registrations.append)

    await BoxdClientManager.get_client()
    await BoxdClientManager.get_client()

    assert len(registrations) == 1


@pytest.mark.asyncio
async def test_close_is_idempotent():
    client = await BoxdClientManager.get_client()

    await BoxdClientManager.close()
    await BoxdClientManager.close()

    assert client.close_calls == 1
    assert BoxdClientManager._client is None


def test_missing_sdk_raises_clear_error(monkeypatch):
    monkeypatch.setattr(client_module, "AsyncBoxd", None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "boxd", None)

    with pytest.raises(ImportError, match="uv sync --extra sandbox"):
        client_module._load_boxd_sdk()
