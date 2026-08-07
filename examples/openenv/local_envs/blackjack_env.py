# SPDX-License-Identifier: Apache-2.0
"""In-process BlackJack env implementing the OpenEnv async-client protocol.

Zero network, zero Docker, zero HF Spaces. Written to be a drop-in target for
`OpenEnvWorkflow` so RL training runs entirely offline against a local model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class _StepResult:
    """Mirrors ``openenv.core.client_types.StepResult``."""

    observation: Any
    reward: float | None = None
    done: bool = False
    metadata: dict[str, Any] | None = None


class LocalBlackjackEnv:
    """A minimal BlackJack env compatible with the OpenEnv async client contract.

    Contract implemented:
      * ``async with env: ...``  → ``__aenter__`` / ``__aexit__``
      * ``await env.reset(seed=int)`` → StepResult with a text observation
      * ``await env.step(action)``    → StepResult with reward + done

    Actions are dicts of the form ``{"action_id": 0}`` (HIT) or
    ``{"action_id": 1}`` (STAND) — the same shape ``OpenSpielEnv`` uses so the
    workflow YAML stays interchangeable.

    Dealer plays hit-on-16, stand-on-17. Ace is 11 unless it would bust.
    """

    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Any = None,
    ) -> None:
        # The base_url / provider / timeouts args are ignored — they exist so
        # the class matches the EnvClient signature the workflow builder uses.
        self._rng: random.Random | None = None
        self._player: list[int] = []
        self._dealer_hidden: int = 0

    # -- OpenEnv async surface ---------------------------------------------

    async def __aenter__(self) -> LocalBlackjackEnv:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def reset(self, *, seed: int | None = None, **_: Any) -> _StepResult:
        self._rng = random.Random(seed)
        self._player = [self._draw(), self._draw()]
        self._dealer_hidden = self._draw()
        return _StepResult(observation=self._observation(), reward=None, done=False)

    async def step(self, action: Any, **_: Any) -> _StepResult:
        action_id = self._action_id(action)
        if action_id == 0:  # HIT
            self._player.append(self._draw())
            if self._hand(self._player) > 21:
                return _StepResult(
                    observation=self._observation(reveal=True),
                    reward=-1.0,
                    done=True,
                    metadata={"reason": "player-bust"},
                )
            return _StepResult(observation=self._observation(), reward=0.0, done=False)

        if action_id == 1:  # STAND -> dealer plays out
            dealer = [self._dealer_hidden, self._draw()]
            while self._hand(dealer) < 17:
                dealer.append(self._draw())
            player_total = self._hand(self._player)
            dealer_total = self._hand(dealer)
            if dealer_total > 21 or player_total > dealer_total:
                reward = 1.0
            elif player_total == dealer_total:
                reward = 0.0
            else:
                reward = -1.0
            return _StepResult(
                observation={
                    "player": list(self._player),
                    "dealer": dealer,
                    "player_total": player_total,
                    "dealer_total": dealer_total,
                },
                reward=reward,
                done=True,
                metadata={"reason": "dealer-plays-out"},
            )

        # Any other action is illegal.
        return _StepResult(
            observation=self._observation(),
            reward=-1.0,
            done=True,
            metadata={"reason": "illegal-action", "action_id": action_id},
        )

    # -- helpers ----------------------------------------------------------

    def _draw(self) -> int:
        assert self._rng is not None
        # 1..10 with face cards = 10; ace low here, workflow can treat 1 as 11.
        return self._rng.randint(1, 10)

    @staticmethod
    def _hand(cards: list[int]) -> int:
        total = sum(cards)
        aces = cards.count(1)
        # Promote aces from 1 to 11 while it doesn't bust.
        while aces and total + 10 <= 21:
            total += 10
            aces -= 1
        return total

    @staticmethod
    def _action_id(action: Any) -> int:
        if isinstance(action, dict):
            return int(action.get("action_id", -1))
        if isinstance(action, int):
            return action
        return -1

    def _observation(self, reveal: bool = False) -> dict[str, Any]:
        return {
            "player": list(self._player),
            "player_total": self._hand(self._player),
            "dealer_showing": self._dealer_hidden if reveal else "hidden",
            "legal_actions": [0, 1],
            "hint": "0=HIT, 1=STAND",
        }
