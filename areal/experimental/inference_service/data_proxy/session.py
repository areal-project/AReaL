# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle management for the data proxy."""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    """Request to start one or more offline RL sessions.

    When ``group_size`` is greater than 1, the data proxy creates multiple
    sessions atomically. The response always contains a flat ``sessions``
    list — single-session is just ``group_size=1``.
    """

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)
    group_size: int = 1


class SessionCredentials(BaseModel):
    """One session's identity and authentication key."""

    session_id: str
    session_api_key: str


class StartSessionResponse(BaseModel):
    """Response from start_session — always a list of session credentials."""

    group_id: str
    sessions: list[SessionCredentials]


class SetRewardRequest(BaseModel):
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float
    model: str | None = None


class ExportTrajectoriesRequest(BaseModel):
    """Request to export trajectories for one or more sessions.

    All sessions are exported and merged into a single interactions dict.
    ``group_id`` is optional metadata used by the gateway for router cleanup;
    the data proxy itself does not use it.
    """

    session_ids: list[str]
    group_id: str | None = None
    trajectory_id: int | None = None
    discount: float = 1.0
    style: str = "individual"
    remove_session: bool = True


class ExportTrajectoriesResponse(BaseModel):
    """Response containing merged serialized interactions."""

    interactions: Any


@dataclass(frozen=True)
class RewardResult:
    """Internal result returned when an online session closes a trajectory."""

    session_id: str
    trajectory_id: int | None
    interaction_count: int
    ready_transition: bool


@dataclass(frozen=True)
class ReadyNotification:
    session_id: str
    trajectory_id: int


@dataclass
class ReadyTrajectory:
    """One ready-but-not-yet-exported online trajectory."""

    trajectory_id: int
    interaction_id: str
    completions: InteractionCache
    created_at: float
    needs_online_callback: bool = False
    callback_delivered: bool = False


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    """Unified session data for both offline and online modes.

    Maintains ``active_completions`` (the current in-progress interaction
    cache) and ``ready_trajectories`` (reward-bounded, exportable
    trajectories).

    - **Offline**: one session → one trajectory via ``set_reward`` →
      ``export_trajectory``.
    - **Online**: one persistent session → many reward-bounded trajectories
      via repeated ``set_reward`` → ``export_trajectory`` calls.
    """

    def __init__(
        self,
        session_id: str,
        set_reward_finish_timeout: float = 0.0,
    ):
        self.session_id = session_id
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._last_access_time = time.time()
        self._lock = threading.Lock()
        self._active_completions = InteractionCache()
        self._ready_trajectories: OrderedDict[int, ReadyTrajectory] = OrderedDict()
        self._next_trajectory_id = 0
        self._last_set_reward_time: float | None = None
        self._last_reward_interaction_id: str | None = None

    def update_last_access(self) -> None:
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    @property
    def active_completions(self) -> InteractionCache:
        return self._active_completions

    @property
    def has_ready_trajectories(self) -> bool:
        with self._lock:
            return bool(self._ready_trajectories)

    def _latest_ready_trajectory_locked(self) -> ReadyTrajectory | None:
        """Return the latest ready trajectory.

        Caller must already hold ``self._lock``.
        """
        if not self._ready_trajectories:
            return None
        return next(reversed(self._ready_trajectories.values()))

    def _resolve_duplicate_ready_locked(
        self, interaction_id: str | None
    ) -> ReadyTrajectory | None:
        latest_ready = self._latest_ready_trajectory_locked()
        if latest_ready is None:
            return None
        if len(self._active_completions) != 0:
            return None
        resolved_interaction_id = interaction_id or latest_ready.interaction_id
        if resolved_interaction_id == latest_ready.interaction_id:
            return latest_ready
        return None

    def _mark_active_trajectory_ready_locked(
        self,
        now: float,
    ) -> RewardResult:
        completions = self._active_completions
        if len(completions) == 0:
            raise ValueError("No interactions in session")

        resolved_interaction_id = self._last_reward_interaction_id
        if resolved_interaction_id is None:
            raise ValueError("No reward has been set for the active trajectory")

        trajectory_id = self._next_trajectory_id
        self._next_trajectory_id += 1
        ready = ReadyTrajectory(
            trajectory_id=trajectory_id,
            interaction_id=resolved_interaction_id,
            completions=completions,
            created_at=now,
            needs_online_callback=True,
        )
        self._ready_trajectories[trajectory_id] = ready
        self._active_completions = InteractionCache()
        self._last_set_reward_time = None
        self._last_reward_interaction_id = None

        return RewardResult(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            interaction_count=len(completions),
            ready_transition=True,
        )

    def _finalize_if_reward_timeout_elapsed_locked(
        self,
        now: float,
    ) -> RewardResult | None:
        if self._last_set_reward_time is None:
            return None
        if now - self._last_set_reward_time < self._set_reward_finish_timeout:
            return None
        return self._mark_active_trajectory_ready_locked(now)

    def set_reward(
        self,
        interaction_id: str | None,
        reward: float,
    ) -> RewardResult:
        """Record reward for the active trajectory."""
        with self._lock:
            now = time.time()
            self._last_access_time = now

            duplicate_ready = self._resolve_duplicate_ready_locked(interaction_id)
            if duplicate_ready is not None:
                return RewardResult(
                    session_id=self.session_id,
                    trajectory_id=duplicate_ready.trajectory_id,
                    interaction_count=len(duplicate_ready.completions),
                    ready_transition=False,
                )

            completions = self._active_completions
            if len(completions) == 0:
                raise ValueError("No interactions in session")

            resolved_interaction_id = interaction_id or completions.last_interaction_id
            if resolved_interaction_id not in completions:
                raise ValueError(f"Interaction {resolved_interaction_id} not found")

            completions.set_reward(resolved_interaction_id, reward)
            self._last_reward_interaction_id = resolved_interaction_id
            self._last_set_reward_time = now

            ready_result = self._finalize_if_reward_timeout_elapsed_locked(now)
            if ready_result is not None:
                return ready_result

            return RewardResult(
                session_id=self.session_id,
                trajectory_id=None,
                interaction_count=len(completions),
                ready_transition=False,
            )

    def finalize_if_reward_timeout_elapsed(
        self,
        now: float | None = None,
    ) -> RewardResult | None:
        with self._lock:
            return self._finalize_if_reward_timeout_elapsed_locked(now or time.time())

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        with self._lock:
            return [
                ReadyNotification(
                    session_id=self.session_id,
                    trajectory_id=ready.trajectory_id,
                )
                for ready in self._ready_trajectories.values()
                if ready.needs_online_callback and not ready.callback_delivered
            ]

    def mark_online_callback_delivered(self, trajectory_id: int) -> bool:
        with self._lock:
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None or not ready.needs_online_callback:
                return False
            if ready.callback_delivered:
                return True
            ready.callback_delivered = True
            return True

    def add_string_interaction(self, messages: list[dict], response: str) -> str:
        interaction_id = str(uuid.uuid4())
        interaction = InteractionWithTokenLogpReward(
            messages=messages,
            output_message_list=[{"role": "assistant", "content": response}],
        )
        interaction._interaction_id = interaction_id
        self._active_completions[interaction_id] = interaction
        self.update_last_access()
        return interaction_id

    def export_trajectory(
        self,
        discount: float,
        style: str,
        trajectory_id: int | None = None,
    ) -> tuple[int, dict[str, InteractionWithTokenLogpReward]]:
        """Export a ready trajectory.

        Parameters
        ----------
        discount : float
            Reward discount factor passed to
            :pymethod:`InteractionCache.export_interactions`.
        style : str
            Export style (``"individual"`` or ``"concat"``).
        trajectory_id : int | None
            Specific trajectory to export.  When ``None``, the latest
            ready trajectory is exported.

        Returns
        -------
        tuple[int, dict[str, InteractionWithTokenLogpReward]]
            ``(trajectory_id, interactions)``

        Raises
        ------
        KeyError
            If no ready trajectories exist, or the requested
            ``trajectory_id`` is not found.
        """
        with self._lock:
            if not self._ready_trajectories:
                raise KeyError(f"No ready trajectories for session {self.session_id}")

            target_trajectory_id = trajectory_id
            if target_trajectory_id is None:
                target_trajectory_id = next(reversed(self._ready_trajectories))

            ready = self._ready_trajectories.pop(target_trajectory_id, None)
            if ready is None:
                raise KeyError(
                    f"Trajectory {target_trajectory_id} not found for session {self.session_id}"
                )

        interactions = ready.completions.export_interactions(
            style=style,
            reward_discount=discount,
        )
        return ready.trajectory_id, interactions


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    """Thread-safe store for session lifecycle management."""

    def __init__(self, set_reward_finish_timeout: float = 0.0):
        self._sessions: dict[str, SessionData] = {}
        self._api_key_to_session: dict[str, str] = {}
        self._session_to_api_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._capacity: int = 0
        self._admin_api_key: str = "areal-admin-key"
        self._set_reward_finish_timeout = set_reward_finish_timeout

    def set_capacity(self, n: int) -> None:
        with self._lock:
            self._capacity = n

    def set_admin_key(self, key: str) -> None:
        with self._lock:
            self._admin_api_key = key

    @property
    def admin_api_key(self) -> str:
        return self._admin_api_key

    def start_session(
        self, task_id: str, api_key: str | None = None
    ) -> tuple[str, str]:
        """Start a new session, returning (session_id, session_api_key).

        If *api_key* is provided the key is reused (refreshed); otherwise a
        fresh opaque key is generated.
        """
        with self._lock:
            idx = 0
            while f"{task_id}-{idx}" in self._sessions:
                idx += 1
            session_id = f"{task_id}-{idx}"

            if api_key:
                session_api_key = api_key
                existing_sid = self._api_key_to_session.get(session_api_key)
                if existing_sid is not None:
                    existing_session = self._sessions.get(existing_sid)
                    if (
                        existing_session is not None
                        and not existing_session.has_ready_trajectories
                    ):
                        raise ValueError(
                            f"API key is already bound to active session {existing_sid}."
                        )
                    self._remove_api_keys_for_session(existing_sid)
            else:
                session_api_key = secrets.token_urlsafe(32)
                while (
                    session_api_key in self._api_key_to_session
                    or session_api_key == self._admin_api_key
                ):
                    session_api_key = secrets.token_urlsafe(32)

            self._sessions[session_id] = SessionData(
                session_id=session_id,
                set_reward_finish_timeout=self._set_reward_finish_timeout,
            )
            self._api_key_to_session[session_api_key] = session_id
            self._session_to_api_key[session_id] = session_api_key

        return (session_id, session_api_key)

    def get_session_by_api_key(self, api_key: str) -> SessionData | None:
        with self._lock:
            session_id = self._api_key_to_session.get(api_key)
            if session_id is None:
                return None
            return self._sessions.get(session_id)

    def get_or_create_hitl_session(self) -> SessionData:
        """Return the persistent HITL session, creating it if needed."""
        with self._lock:
            session = self._sessions.get("__hitl__")
            if session is None:
                session = SessionData(
                    session_id="__hitl__",
                    set_reward_finish_timeout=self._set_reward_finish_timeout,
                )
                self._sessions["__hitl__"] = session
            return session

    def get_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._remove_api_keys_for_session(session_id)

    def _remove_api_keys_for_session(self, session_id: str) -> None:
        api_key = self._session_to_api_key.pop(session_id, None)
        if api_key:
            self._api_key_to_session.pop(api_key, None)

    def cleanup_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> None:
        with self._lock:
            stale_sessions: list[str] = []
            for sid, session in self._sessions.items():
                if not session.is_stale(timeout_seconds):
                    continue
                if session.has_ready_trajectories:
                    continue
                stale_sessions.append(sid)

            for sid in stale_sessions:
                self._sessions.pop(sid, None)
                self._remove_api_keys_for_session(sid)

    def finalize_rewarded_trajectories(
        self,
        now: float | None = None,
    ) -> list[RewardResult]:
        with self._lock:
            sessions = list(self._sessions.values())

        finalized: list[RewardResult] = []
        resolved_now = time.time() if now is None else now
        for session in sessions:
            ready_result = session.finalize_if_reward_timeout_elapsed(resolved_now)
            if ready_result is not None:
                finalized.append(ready_result)
        return finalized

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        with self._lock:
            sessions = list(self._sessions.values())

        notifications: list[ReadyNotification] = []
        for session in sessions:
            notifications.extend(session.pending_online_callbacks())
        return notifications

    def mark_online_callback_delivered(
        self, session_id: str, trajectory_id: int
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        return session.mark_online_callback_delivered(trajectory_id)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
