"""Async HTTP client for the ALFWorld env server.

Used by workflow.py to communicate with env_server.py running in
a separate container where textworld works.
"""
import os
import logging
import asyncio
from typing import Tuple

import aiohttp

logger = logging.getLogger(__name__)

ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:8765")
TIMEOUT = aiohttp.ClientTimeout(total=60, connect=5)


class EnvClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or ENV_SERVER_URL).rstrip("/")
        self._session: aiohttp.ClientSession = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=TIMEOUT)
        return self._session

    async def _post(self, path: str, data: dict, retries: int = 2) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        for attempt in range(retries + 1):
            try:
                async with session.post(url, json=data) as resp:
                    if resp.status == 503 and attempt < retries:
                        await asyncio.sleep(1.0 + attempt * 2.0)
                        continue
                    if resp.status >= 500 and attempt < retries:
                        await asyncio.sleep(0.5)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue
                raise RuntimeError(f"Env server request failed: {path} - {e}") from e
        return {}

    async def create(self, game_file: str) -> str:
        result = await self._post("/create", {"game_file": game_file})
        return result["env_id"]

    async def reset(self, env_id: str) -> Tuple[str, dict]:
        result = await self._post("/reset", {"env_id": env_id})
        return result["obs"], result.get("info", {})

    async def step(self, env_id: str, action: str) -> Tuple[str, float, bool, bool]:
        result = await self._post("/step", {"env_id": env_id, "action": action})
        if result.get("error"):
            logger.warning("Env step returned error: %s", result["error"])
        return result["obs"], result["reward"], result["done"], result.get("won", False)

    async def close(self, env_id: str):
        try:
            await self._post("/close", {"env_id": env_id})
        except Exception as e:
            logger.warning("Failed to close env %s: %s", env_id, e)

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()


_client: EnvClient = None


def get_client() -> EnvClient:
    global _client
    if _client is None:
        _client = EnvClient()
    return _client
