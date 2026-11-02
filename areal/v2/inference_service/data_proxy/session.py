# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle management for the data proxy."""

from __future__ import annotations

import asyncio
import math
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from areal.experimental.openai.cache import (
    InteractionCache,
    PrefixMatcher,
    copy_interactions,
)
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


class EpisodeIdentity(BaseModel):
    """Caller snapshot fencing delayed requests across episode boundaries."""

    incarnation_id: str = Field(min_length=1)
    trajectory_id: int = Field(ge=0, strict=True)


class SessionCredentials(BaseModel):
    """One session's identity and authentication key."""

    session_id: str
    session_api_key: str
    areal_episode: EpisodeIdentity | None = None


class StartSessionResponse(BaseModel):
    """Response from start_session — always a list of session credentials."""

    group_id: str
    sessions: list[SessionCredentials]


class SetRewardRequest(BaseModel):
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float
    model: str | None = None
    areal_episode: EpisodeIdentity | None = None


class AbortEpisodeRequest(EpisodeIdentity):
    """Cancel a specific unrewarded episode, including lost-admission retries."""

    episode_id: str | None = None
    reason: str = "client_abort"


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
    drop_retry_orphans: bool = False
    reward_normalization: bool = False
    discard_trajectory: bool = False
    consume: bool = True
    acknowledge_only: bool = False
    inline_tensors: bool = False
    episode_id: str | None = None
    lease: str | None = None
    handoff_digest: str | None = None

    include_observation: bool = False


class ExportTrajectoriesResponse(BaseModel):
    """Response containing merged serialized interactions."""

    traj: dict[str, Any]
    observation: dict[str, Any] | None = None


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
    episode_id: str | None = None
    lease: str | None = None


@dataclass
class ReadyTrajectory:
    """One ready-but-not-yet-exported online trajectory."""

    trajectory_id: int
    interaction_id: str
    completions: InteractionCache
    created_at: float
    needs_online_callback: bool = False
    callback_delivered: bool = False
    export_options: tuple[float, str, bool] | None = None
    exported_interactions: dict[str, InteractionWithTokenLogpReward] | None = None
    episode_id: str | None = None
    lease: str | None = None
    export_digest: str | None = None


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
        sampling_seed_identity: str | None = None,
        prefix_matcher: PrefixMatcher | None = None,
    ):
        self.session_id = session_id
        self.incarnation_id = uuid.uuid4().hex
        self.sampling_seed_identity = sampling_seed_identity or session_id
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._prefix_matcher = prefix_matcher
        self._last_access_time = time.time()
        self.created_at = self._last_access_time
        self._lock = threading.Lock()
        self._active_completions = self._new_interaction_cache()
        self._ready_trajectories: OrderedDict[int, ReadyTrajectory] = OrderedDict()
        self._next_trajectory_id = 0
        self._acknowledged_trajectories: set[int] = set()
        self._acknowledged_episode_leases: dict[int, tuple[str | None, str | None]] = {}
        self._acknowledged_digests: dict[int, str | None] = {}
        self._next_sampling_request_index = 0
        self._last_set_reward_time: float | None = None
        self._last_reward_interaction_id: str | None = None
        self._admission_lock = asyncio.Lock()
        self._active_episode: dict[str, str] | None = None
        self._admission_attempted = False
        self._generation_uncertain = False
        self._aborting_episode: dict[str, Any] | None = None
        self._aborted_episodes: dict[int, dict[str, Any]] = {}
        self._reward_receipts: dict[int, tuple[str, float, int]] = {}
        self._inflight_generations = 0
        self._online_notifications: dict[int, ReadyNotification] = {}

    async def begin_generation(
        self,
        admit: Callable[[str, int], Awaitable[dict[str, Any]]] | None = None,
        *,
        episode: EpisodeIdentity | None = None,
    ) -> InteractionCache:
        """Concurrent tool calls share one credit and capture one active cache."""
        # Check before waiting as well: a request queued behind an abort must
        # never silently become the first request of the next episode.
        with self._lock:
            self._check_generation_locked(episode, require_identity=admit is not None)
        async with self._admission_lock:
            with self._lock:
                self._check_generation_locked(
                    episode, require_identity=admit is not None
                )
                trajectory_id = self._next_trajectory_id
                needs_credit = admit is not None and self._active_episode is None
                if needs_credit:
                    # A failed admission response may still have allocated a
                    # credit. Preserve this intent until explicit cancellation.
                    self._admission_attempted = True
            if needs_credit:
                assert admit is not None
                admission = await admit(self.session_id, trajectory_id)
                with self._lock:
                    self._active_episode = admission
            with self._lock:
                self._inflight_generations += 1
                return self._active_completions

    def _check_generation_locked(
        self, episode: EpisodeIdentity | None, *, require_identity: bool
    ) -> None:
        if self._aborting_episode is not None:
            raise ValueError("Episode abort is awaiting durable confirmation")
        if self._generation_uncertain:
            raise ValueError("A generation outcome is unknown; episode is quarantined")
        if require_identity:
            self._check_incarnation_locked(episode)
            if episode.trajectory_id != self._next_trajectory_id:
                raise ValueError(
                    "Request targets a different or already sealed episode"
                )

    def _check_incarnation_locked(self, episode: EpisodeIdentity | None) -> None:
        if episode is None or episode.incarnation_id != self.incarnation_id:
            raise ValueError("areal_episode must identify this session incarnation")

    def finish_generation(self, *, uncertain: bool = False) -> None:
        with self._lock:
            if self._inflight_generations < 1:
                raise RuntimeError("Generation completion has no active request")
            self._inflight_generations -= 1
            if uncertain and self._active_episode is not None:
                self._generation_uncertain = True

    @property
    def has_online_episode(self) -> bool:
        with self._lock:
            return self._admission_attempted or self._active_episode is not None

    def current_episode(self) -> dict[str, Any]:
        with self._lock:
            return {
                "incarnation_id": self.incarnation_id,
                "trajectory_id": self._next_trajectory_id,
                "episode_id": self._active_episode["episode_id"]
                if self._active_episode
                else None,
                "inflight_generations": self._inflight_generations,
                "state": (
                    "quarantined"
                    if self._generation_uncertain
                    else "aborting"
                    if self._aborting_episode is not None
                    else "active"
                    if self._active_episode is not None
                    else "admission_unconfirmed"
                    if self._admission_attempted
                    else "idle"
                ),
            }

    async def abort_episode(
        self,
        request: AbortEpisodeRequest,
        cancel: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Fence new work, durably retire its credit, then advance the session.

        An unknown backend outcome cannot be cancelled by this interface. An
        admission timeout is safe: no generation was dispatched, and the CPU
        ledger creates a tombstone before any delayed admit can take effect.
        """
        async with self._admission_lock:
            with self._lock:
                if (
                    request.incarnation_id != self.incarnation_id
                    or type(request.trajectory_id) is not int
                    or request.trajectory_id < 0
                ):
                    raise ValueError("Episode does not match this session incarnation")
                previous = self._aborted_episodes.get(request.trajectory_id)
                if previous is not None:
                    if request.episode_id not in (
                        previous["requested_episode_id"],
                        previous["episode_id"],
                    ):
                        raise ValueError("Abort retry targets a different episode")
                    return {
                        key: previous[key]
                        for key in ("episode_id", "trajectory_id", "status")
                    }
                episode = self._active_episode
                if (
                    request.trajectory_id != self._next_trajectory_id
                    or request.episode_id
                    != (episode["episode_id"] if episode else None)
                ):
                    raise ValueError(
                        "Abort targets a different or already sealed episode"
                    )
                if (
                    not self._admission_attempted
                    or self._inflight_generations
                    or self._generation_uncertain
                ):
                    raise ValueError(
                        "Abort requires an admitted episode with no in-flight or unknown generation"
                    )
                if self._last_set_reward_time is not None:
                    raise ValueError(
                        "A rewarded episode must finish its trajectory handoff"
                    )
                if not request.reason.strip() or len(request.reason) > 256:
                    raise ValueError(
                        "Abort requires a nonempty reason of at most 256 characters"
                    )
                if self._aborting_episode is None:
                    self._aborting_episode = {
                        "session_id": self.session_id,
                        "trajectory_id": request.trajectory_id,
                        "episode_id": request.episode_id,
                        "lease": episode["lease"] if episode else None,
                        "reason": request.reason,
                    }
                intent = dict(self._aborting_episode)
            # Retain the closing fence on errors, cancellation and lost replies.
            result = await cancel(intent)
            if (
                result.get("status") != "rejected"
                or not result.get("episode_id")
                or intent["episode_id"] is not None
                and result["episode_id"] != intent["episode_id"]
            ):
                raise RuntimeError("Episode cancellation was not durably confirmed")
            with self._lock:
                receipt = {
                    "episode_id": result["episode_id"],
                    "trajectory_id": request.trajectory_id,
                    "requested_episode_id": request.episode_id,
                    "status": "aborted",
                }
                self._aborted_episodes[request.trajectory_id] = receipt
                self._active_completions = self._new_interaction_cache()
                self._active_episode = None
                self._admission_attempted = False
                self._aborting_episode = None
                self._next_trajectory_id += 1
                self._last_access_time = time.time()
                return {
                    key: receipt[key]
                    for key in ("episode_id", "trajectory_id", "status")
                }

    def next_sampling_request_index(self) -> int:
        """Reserve a request index before inference without serializing generation."""
        with self._lock:
            request_index = self._next_sampling_request_index
            self._next_sampling_request_index += 1
        return request_index

    def _new_interaction_cache(self) -> InteractionCache:
        return InteractionCache(
            session_id=self.session_id,
            prefix_matcher=self._prefix_matcher,
        )

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
            episode_id=self._active_episode["episode_id"]
            if self._active_episode
            else None,
            lease=self._active_episode["lease"] if self._active_episode else None,
        )
        self._ready_trajectories[trajectory_id] = ready
        if ready.episode_id is not None:
            self._reward_receipts[trajectory_id] = (
                resolved_interaction_id,
                completions[resolved_interaction_id].reward,
                len(completions),
            )
        if ready.episode_id is not None:
            # A lost ack response must not erase the only retry trigger. Keep
            # the notification until the receiver confirms the whole handoff.
            self._online_notifications[trajectory_id] = ReadyNotification(
                self.session_id, trajectory_id, ready.episode_id, ready.lease
            )
        self._active_completions = self._new_interaction_cache()
        self._active_episode = None
        self._admission_attempted = False
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
        if (
            self._last_set_reward_time is None
            or self._inflight_generations
            or self._generation_uncertain
            or self._aborting_episode is not None
        ):
            return None
        if now - self._last_set_reward_time < self._set_reward_finish_timeout:
            return None
        return self._mark_active_trajectory_ready_locked(now)

    def set_reward(
        self,
        interaction_id: str | None,
        reward: float,
        *,
        episode: EpisodeIdentity | None = None,
        require_episode_identity: bool = False,
    ) -> RewardResult:
        """Record reward for the active trajectory."""
        with self._lock:
            if require_episode_identity:
                self._check_incarnation_locked(episode)
                if not math.isfinite(reward):
                    raise ValueError("Online episode reward must be finite")
                if episode.trajectory_id != self._next_trajectory_id:
                    receipt = self._reward_receipts.get(episode.trajectory_id)
                    if receipt is None or (
                        interaction_id not in (None, receipt[0]) or reward != receipt[1]
                    ):
                        raise ValueError(
                            "Reward targets a different or already sealed episode"
                        )
                    return RewardResult(
                        self.session_id, episode.trajectory_id, receipt[2], False
                    )
            now = time.time()
            self._last_access_time = now
            if self._generation_uncertain or self._aborting_episode is not None:
                raise ValueError(
                    "The active episode is quarantined or awaiting abort confirmation"
                )

            if require_episode_identity and self._active_episode is None:
                raise ValueError("No active admitted episode to reward")
            duplicate_ready = (
                None
                if require_episode_identity
                else self._resolve_duplicate_ready_locked(interaction_id)
            )
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
            return list(self._online_notifications.values()) + [
                ReadyNotification(
                    session_id=self.session_id,
                    trajectory_id=ready.trajectory_id,
                    episode_id=ready.episode_id,
                    lease=ready.lease,
                )
                for ready in self._ready_trajectories.values()
                if ready.needs_online_callback
                and not ready.callback_delivered
                and ready.episode_id is None
            ]

    def mark_online_callback_delivered(self, trajectory_id: int) -> bool:
        with self._lock:
            notification = self._online_notifications.pop(trajectory_id, None)
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None or not ready.needs_online_callback:
                return notification is not None
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
        drop_retry_orphans: bool = False,
        consume: bool = True,
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
        drop_retry_orphans : bool
            Remove duplicate responses left by upstream request retries before
            reward discounting and concat export.

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

            ready = self._ready_trajectories.get(target_trajectory_id)
            if ready is None:
                raise KeyError(
                    f"Trajectory {target_trajectory_id} not found for session {self.session_id}"
                )

            options = (discount, style, drop_retry_orphans)
            if ready.export_options is not None and ready.export_options != options:
                raise ValueError(
                    "A retained trajectory must use the same export options"
                )
            if ready.exported_interactions is None:
                interactions = ready.completions.snapshot().export_interactions(
                    style=style,
                    reward_discount=discount,
                    drop_retry_orphans=drop_retry_orphans,
                )
                ready.export_options = options
                ready.exported_interactions = interactions
            if consume:
                self._ready_trajectories.pop(target_trajectory_id)
                self._acknowledged_trajectories.add(target_trajectory_id)
                self._acknowledged_episode_leases[target_trajectory_id] = (
                    ready.episode_id,
                    ready.lease,
                )
            return ready.trajectory_id, copy_interactions(ready.exported_interactions)

    def trajectory_observation_times(self, trajectory_id: int) -> tuple[float, float]:
        """Read timestamps without exporting or consuming trajectory data."""
        with self._lock:
            return self.created_at, self._ready_trajectories[trajectory_id].created_at

    def validate_trajectory_identity(
        self,
        trajectory_id: int | None,
        episode_id: str | None,
        lease: str | None,
    ) -> None:
        """Fence delayed export/ack calls from a previous session incarnation."""
        with self._lock:
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None and trajectory_id is None and self._ready_trajectories:
                ready = next(reversed(self._ready_trajectories.values()))
            expected = (
                (ready.episode_id, ready.lease)
                if ready is not None
                else self._acknowledged_episode_leases.get(trajectory_id, (None, None))
            )
            if expected != (episode_id, lease):
                raise ValueError(
                    "Trajectory does not match this episode admission lease"
                )

    def record_export_digest(self, trajectory_id: int, digest: str) -> None:
        with self._lock:
            ready = self._ready_trajectories[trajectory_id]
            if ready.episode_id is not None:
                if ready.export_digest is not None and ready.export_digest != digest:
                    raise ValueError("Retained export payload changed")
                ready.export_digest = digest

    def acknowledge_trajectory(
        self,
        trajectory_id: int,
        handoff_digest: str | None = None,
    ) -> None:
        """Retire an exported trajectory; retries never retire a different one."""
        with self._lock:
            if trajectory_id in self._acknowledged_trajectories:
                expected = self._acknowledged_digests.get(trajectory_id)
                if expected is not None and expected != handoff_digest:
                    raise ValueError(
                        "Acknowledgement does not match the durable receipt"
                    )
                return
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None:
                raise KeyError(f"Trajectory {trajectory_id} not found")
            if ready.exported_interactions is None:
                raise ValueError("Cannot acknowledge a trajectory before exporting it")
            if ready.episode_id is not None and (
                ready.export_digest is None or handoff_digest != ready.export_digest
            ):
                raise ValueError("Acknowledgement does not match the durable receipt")
            self._ready_trajectories.pop(trajectory_id)
            self._acknowledged_trajectories.add(trajectory_id)
            self._acknowledged_episode_leases[trajectory_id] = (
                ready.episode_id,
                ready.lease,
            )
            self._acknowledged_digests[trajectory_id] = ready.export_digest


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    """Thread-safe store for session lifecycle management."""

    def __init__(
        self,
        set_reward_finish_timeout: float = 0.0,
        prefix_matcher: PrefixMatcher | None = None,
    ):
        self._sessions: dict[str, SessionData] = {}
        self._api_key_to_session: dict[str, str] = {}
        self._session_to_api_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._capacity: int = 0
        self._admin_api_key: str = "areal-admin-key"
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._prefix_matcher = prefix_matcher

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
        self,
        task_id: str,
        api_key: str | None = None,
        sampling_seed_identity: str | None = None,
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
                    if existing_session is not None and (
                        existing_session.has_online_episode
                        or not existing_session.has_ready_trajectories
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
                sampling_seed_identity=sampling_seed_identity,
                prefix_matcher=self._prefix_matcher,
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
                    prefix_matcher=self._prefix_matcher,
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
                if (
                    session.has_online_episode
                    or session.has_ready_trajectories
                    or session.pending_online_callbacks()
                ):
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
