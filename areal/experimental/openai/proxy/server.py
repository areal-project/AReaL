# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    """Request to start a new RL session."""

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)


class StartSessionResponse(BaseModel):
    """Response from start_session endpoint."""

    session_id: str
    api_key: str


class SetRewardRequest(BaseModel):
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float


class ExportTrajectoriesRequest(BaseModel):
    """Request to export trajectories for a session."""

    session_id: str
    discount: float = 1.0
    style: str = "individual"


class ExportTrajectoriesResponse(BaseModel):
    """Response containing serialized interactions."""

    interactions: dict[str, Any]


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    """Data associated with a single RL session."""

    def __init__(self, session_id: str):
        self.session_id = session_id

        self._completed = False
        self._completions = InteractionCache()
        self._completed_event = threading.Event()
        self._start_time = time.time()
        self._last_access_time = time.time()
        self._end_time = None
        self._lock = threading.Lock()

    def update_last_access(self):
        """Update the last access time for this session."""
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        """Check if this session has been inactive for too long."""
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    def finish(self):
        self._completed = True
        self._end_time = time.time()
        self._completed_event.set()

    @property
    def is_completed(self) -> bool:
        """Whether this session has been completed via ``finish()``."""
        return self._completed

    @property
    def completions(self):
        return self._completions

    async def wait_for_finish(self, timeout: float | None = None) -> bool:
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout if timeout else None
        while not self._completed_event.is_set():
            remaining = (deadline - time.monotonic()) if deadline else 1.0
            if deadline and remaining <= 0:
                return False
            poll = min(remaining, 1.0)  # Poll every 1s so cancellation works
            await loop.run_in_executor(None, self._completed_event.wait, poll)
        return True

    def export_interactions(
        self, discount: float, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        if len(self.completions) == 0:
            return {}
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


# =============================================================================
# Serialization Helpers
# =============================================================================


def serialize_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, Any]:
    """Serialize interactions into a json-compatible format for HTTP transport."""
    from areal.infra.rpc.serialization import serialize_value

    result = {}
    for key, interaction in interactions.items():
        if interaction.has_tensor_data:
            result[key] = {
                "tensor_dict": interaction.to_tensor_dict(),
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
        else:
            result[key] = {
                "messages": interaction.messages,
                "output_message_list": interaction.output_message_list,
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
    return serialize_value(result)


def deserialize_interactions(
    data: dict[str, Any],
) -> dict[str, InteractionWithTokenLogpReward]:
    """Deserialize interactions from HTTP response."""
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.infra.rpc.serialization import deserialize_value

    data = deserialize_value(data)
    result = {}
    for key, item in data.items():
        interaction = InteractionWithTokenLogpReward()
        if "tensor_dict" in item:
            interaction._cache = item["tensor_dict"]
        else:
            interaction.messages = item["messages"]
            interaction.output_message_list = item["output_message_list"]
        interaction.reward = item["reward"]
        interaction.interaction_id = item["interaction_id"]
        result[key] = interaction
    return result


# =============================================================================
# Path Constants (must match client_session.py expectations)
# =============================================================================

RL_START_SESSION_PATHNAME = "rl/start_session"
RL_END_SESSION_PATHNAME = "rl/end_session"
RL_SET_REWARD_PATHNAME = "rl/set_reward"
CHAT_COMPLETIONS_PATHNAME = "chat/completions"
RESPONSES_PATHNAME = "responses"
ANTHROPIC_MESSAGES_PATHNAME = "v1/messages"
GRANT_CAPACITY_PATHNAME = "grant_capacity"
EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"
INTERNAL_WAIT_FOR_SESSION_PATHNAME = "internal/wait_for_session"

# Shared default for admin API key — used by cli_args.py and workflow.py
# to avoid independent duplication.
DEFAULT_ADMIN_API_KEY = "areal-admin-key"


class WaitForSessionRequest(BaseModel):
    """Request from _OnlineAgent to register a worker and wait for a session."""

    worker_addr: str


class WaitForSessionResponse(BaseModel):
    """Response with completed session credentials."""

    session_api_key: str
    session_id: str
    worker_addr: str
