# SPDX-License-Identifier: Apache-2.0
"""Config dataclasses and Protocols for the OpenEnv adapter.

OpenEnv (https://github.com/huggingface/OpenEnv) exposes agentic environments
via a uniform reset/step/state HTTP+WebSocket surface. This module defines the
configuration surface for AReaL's ``OpenEnvWorkflow`` and the pluggable
adapter Protocols (action parser + observation formatter) that let a workflow
target arbitrary OpenEnv environments without new code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ObservationFormatter(Protocol):
    """Convert an OpenEnv observation into a chat message dict.

    Implementations must be stateless with respect to a single episode; per-step
    state should live on ``OpenEnvWorkflow`` or in the observation itself.
    """

    def __call__(self, observation: Any, step: int) -> dict[str, str]:
        """Return a ``{"role": ..., "content": ...}`` message for the LLM."""
        ...


@runtime_checkable
class ActionParser(Protocol):
    """Convert an LLM completion string into an OpenEnv action object.

    Returning ``None`` marks the completion as unparsable and the workflow
    treats the step as a failed no-op with zero reward.
    """

    def __call__(self, completion: str, observation: Any) -> Any:
        """Return the parsed action, or ``None`` on parse failure."""
        ...


@dataclass
class OpenEnvConfig:
    """Configuration for :class:`areal.workflow.openenv.OpenEnvWorkflow`.

    Attributes:
        env_client_class: Fully-qualified import path to an
            :class:`openenv.core.env_client.EnvClient` subclass or a factory
            callable that returns one. Example: ``"echo_env.EchoEnv"``.
        base_url: Base URL of a running environment server (``http://`` or
            ``ws://``). Mutually exclusive with ``provider``. Set exactly one.
        provider: Provider spec for launching the environment locally. One of
            ``"uv"`` (uses ``UVProvider``, no Docker) or ``"docker"``
            (``LocalDockerProvider``). Leave unset to require ``base_url``.
        project_path: Project source for ``provider="uv"``. Either a local path
            or a ``git+<url>`` spec, forwarded to ``UVProvider``.
        docker_image: Image tag for ``provider="docker"``. Ignored otherwise.
        action_class: Optional fully-qualified import path to the Action
            dataclass expected by ``env.step`` (e.g. ``"echo_env.CallToolAction"``).
            When set, ``ActionParser`` output is passed as ``action_class(**parsed)``
            if the parser returns a dict; otherwise passed through as-is.
        action_parser: Import path to an :class:`ActionParser` implementation,
            or a shorthand: ``"json"`` (default), ``"tag"``
            (``<action>...</action>``), or ``"passthrough"`` (raw string).
        obs_formatter: Import path to an :class:`ObservationFormatter`, or the
            shorthand ``"auto"`` (dataclass/dict → JSON, str → identity).
        system_prompt: Optional system message prepended to every episode.
        max_turns: Hard cap on env steps per episode.
        step_discount: Per-step reward discount (``reward *= step_discount``
            before accumulation). ``1.0`` disables discounting.
        terminal_reward_only: When ``True``, only the last step's reward is
            kept; intermediate rewards are discarded (episode-level scoring).
        reset_kwargs: Keyword args forwarded to ``env.reset`` (e.g. ``{"seed": 0}``).
        connect_timeout_s: WebSocket connect timeout.
        message_timeout_s: WebSocket message timeout.
    """

    env_client_class: str
    base_url: str | None = None
    provider: str | None = None
    project_path: str | None = None
    docker_image: str | None = None
    action_class: str | None = None
    action_parser: str = "json"
    obs_formatter: str = "auto"
    system_prompt: str = ""
    max_turns: int = 8
    step_discount: float = 1.0
    terminal_reward_only: bool = False
    reset_kwargs: dict[str, Any] = field(default_factory=dict)
    connect_timeout_s: float = 30.0
    message_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")
        if not 0.0 < self.step_discount <= 1.0:
            raise ValueError(
                f"step_discount must be in (0, 1], got {self.step_discount}"
            )
        if self.base_url is None and self.provider is None:
            raise ValueError(
                "OpenEnvConfig requires either base_url or provider to be set."
            )
        if self.base_url is not None and self.provider is not None:
            raise ValueError(
                "OpenEnvConfig.base_url and OpenEnvConfig.provider are "
                "mutually exclusive; set exactly one."
            )
        if self.provider is not None and self.provider not in ("uv", "docker"):
            raise ValueError(
                f"OpenEnvConfig.provider must be 'uv' or 'docker', "
                f"got {self.provider!r}"
            )
        if self.provider == "uv" and self.project_path is None:
            raise ValueError(
                "OpenEnvConfig.provider='uv' requires project_path to be set."
            )
        if self.provider == "docker" and self.docker_image is None:
            raise ValueError(
                "OpenEnvConfig.provider='docker' requires docker_image to be set."
            )


__all__ = [
    "ActionParser",
    "ObservationFormatter",
    "OpenEnvConfig",
]
