"""Unit tests for _OnlineAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.experimental.openai.proxy.online_agent import _OnlineAgent
from areal.experimental.openai.proxy.proxy_gateway import CompletedSessionInfo


class TestOnlineAgentInit:
    def test_init_stores_params(self):
        agent = _OnlineAgent(
            proxy_gateway_addr="http://ctrl:9000",
            admin_api_key="sk-admin",
            timeout=60.0,
        )
        assert agent.proxy_gateway_addr == "http://ctrl:9000"
        assert agent.admin_api_key == "sk-admin"
        assert agent.timeout == 60.0

    def test_init_default_timeout(self):
        agent = _OnlineAgent(
            proxy_gateway_addr="http://ctrl:9000",
            admin_api_key="sk-admin",
        )
        assert agent.timeout == 3600.0


class TestOnlineAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_completed_session_info(self):
        """Test that _OnlineAgent.run() returns CompletedSessionInfo on success.

        Mocks aiohttp.ClientSession to avoid real HTTP calls.
        """
        expected = {
            "session_api_key": "sk-sess-123",
            "session_id": "sess-abc",
            "worker_addr": "http://worker-0:8000",
        }

        # Build mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        agent = _OnlineAgent(
            proxy_gateway_addr="http://ctrl:9000",
            admin_api_key="sk-admin",
            timeout=10.0,
        )

        with patch(
            "areal.experimental.openai.proxy.online_agent.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await agent.run(
                data={},
                base_url="http://worker-0:8000",
                api_key="sk-admin",
                http_client=None,
            )

        assert isinstance(result, CompletedSessionInfo)
        assert result.session_api_key == "sk-sess-123"
        assert result.session_id == "sess-abc"
        assert result.worker_addr == "http://worker-0:8000"

        # Verify the correct URL and payload were sent
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "internal/wait_for_session" in call_args[0][0]
        assert call_args[1]["json"] == {"worker_addr": "http://worker-0:8000"}

    @pytest.mark.asyncio
    async def test_run_raises_on_server_error(self):
        """Test that _OnlineAgent.run() raises on HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("Server error 500")
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        agent = _OnlineAgent(
            proxy_gateway_addr="http://ctrl:9000",
            admin_api_key="sk-admin",
            timeout=10.0,
        )

        with patch(
            "areal.experimental.openai.proxy.online_agent.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(Exception, match="Server error 500"):
                await agent.run(
                    data={},
                    base_url="http://worker-0:8000",
                    api_key="sk-admin",
                    http_client=None,
                )
