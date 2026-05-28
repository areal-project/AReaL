# SPDX-License-Identifier: Apache-2.0

import threading
from collections import OrderedDict
from typing import Any

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("OpenAICache")


class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_reward_discount_called = False
        self._total_reward = 0.0
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        """Allow deep-copy of the empty cache.

        ``threading.Lock`` cannot be deep-copied.  Controllers that hold
        an ``InteractionCache`` (e.g. ``ChatTracer``) are cloned via
        ``Controller.clone()`` (``copy.deepcopy``).  The cache must be
        empty at clone time; a non-empty cache indicates a bug in the
        caller.
        """
        assert len(self) == 0, (
            f"InteractionCache must be empty when deep-copied, but has {len(self)} items"
        )
        new = InteractionCache()
        memo[id(self)] = new
        return new

    @property
    def last_interaction_id(self) -> str:
        return next(reversed(self))

    @property
    def total_reward(self) -> float:
        return self._total_reward

    def set_reward(self, interaction_id: str, reward: float) -> None:
        """Set reward for a specific completion/response by its ID."""
        with self._lock:  # usually no need to lock, but just in case
            self._total_reward -= self[interaction_id].reward or 0.0
            self[interaction_id].reward = reward
            self._total_reward += reward

    def set_last_reward(self, reward: float) -> None:
        """Set reward for the most recent completion/response."""
        self.set_reward(self.last_interaction_id, reward)

    def apply_reward_discount(
        self, turn_discount: float = 1.0
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Apply backward discounted rewards across cached completions/responses.

        This method iterates over the cached completions/responses in reverse creation
        (insertion) order and applies a geometric discount to propagate reward
        signal backward in time. The most recent completion/response is treated as the
        starting point. If it does not have an explicit reward, a warning is
        logged and a default reward of ``0.0`` is used. For each earlier
        completion/response, its reward is initialized to ``0.0`` if unset, then the
        discounted reward from the next later completion/response is added:

        ``reward[i] += reward[i+1] * turn_discount``.

        Typically called before exporting completions/responses in 'individual' style
        to each completion/response is assigned with a valid reward value.

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion/response to an earlier one, by default 1.0.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A shallow copy of the completion/response cache after rewards have been
            updated in-place.
        """
        # Assign rewards to interactions in cache based on their creation order
        if self._apply_reward_discount_called:
            raise RuntimeError("apply_reward_discount should only be called once.")
        self._apply_reward_discount_called = True
        reversed_interactions = list(reversed(self.values()))

        if reversed_interactions:
            current_reward = 0.0
            for i, interaction in enumerate(reversed_interactions):
                if interaction.reward is None:
                    # If the last-created interaction has no reward set, log a warning
                    if i == 0:
                        logger.warning(
                            "The most recent interaction does not have a reward set. "
                            "All interactions will have None reward."
                        )
                    interaction.reward = 0.0

                current_reward = current_reward * turn_discount + interaction.reward
                interaction.reward = current_reward
        return dict(**self)

    def __setitem__(
        self,
        key: str,
        value: InteractionWithTokenLogpReward,
    ) -> None:
        """Add a new interaction to the cache, automatically building parent-child relationships."""
        if value.messages is None:
            raise ValueError(
                "Interaction messages must be set to find parent relationship."
            )

        def _is_prefix(a: list[dict], b: list[dict]) -> bool:
            # True if a is a prefix of b
            if len(a) > len(b):
                return False
            return b[: len(a)] == a

        def _is_similar_on_last_message(
            a: list[dict], b: list[dict]
        ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
            if len(a) > len(b):
                return False, None, None
            last_a_message = a[-1]
            last_b_message = b[len(a) - 1]

            same_keys = set(last_a_message.keys()).intersection(
                set(last_b_message.keys())
            )
            for key in same_keys:
                if last_a_message[key] != last_b_message[key]:
                    return False, None, None
            diff_a_message = {
                k: v for k, v in last_a_message.items() if k not in same_keys
            }
            diff_b_message = {
                k: v for k, v in last_b_message.items() if k not in same_keys
            }
            return True, diff_a_message, diff_b_message

        # Construct parent-child relationships using longest prefix rule
        # Sort potential parents by message length to find the longest prefix match first.
        interactions = sorted(
            self.values(), key=lambda x: len(x.messages), reverse=True
        )

        # Find parent for the new interaction
        for parent in interactions:
            # Skip interactions that are still being processed (output_message_list not set yet)
            # This can happen with concurrent requests where a streaming request hasn't
            # finished setting up yet. Such interactions cannot be parents anyway.
            if parent.output_message_list is None or parent.messages is None:
                continue
            parent_data = parent.messages + parent.output_message_list
            if _is_prefix(parent_data, value.messages):
                value.parent = parent
                break
            elif _is_prefix(parent.messages, value.messages):
                is_similar, diff_a, diff_b = _is_similar_on_last_message(
                    parent_data, value.messages
                )
                if is_similar:
                    logger.warning(
                        "Found a parent interaction with similar last message content, "
                        "but not a strict prefix match. If you wish to use concat mode and build a conversation tree:\n"
                        "1. For completion, append `chat_completion.choices[0].message.model_dump()` to your messages.\n"
                        "2. For response, extend `[o.model_dump() for o in response.output]` to your messages.\n"
                        f"Different keys in parent last message: {diff_a}\n"
                        f"Different keys in child last message: {diff_b}\n"
                    )
        super().__setitem__(key, value)

    def export_interactions(
        self, style: str, reward_discount: float | None = None
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Export cached completions/responses in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions/responses whose input message lists form a strict-prefix
        relationship. The longest-prefix rule is used to determine each node's
        parent. It then returns only leaf-node completions/responses (those without
        children). No reward propagation is performed here.

        When ``style='individual'``, all cached completions/responses are returned as-is
        without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'concat'.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A mapping from completion/response ID to completion/response objects. For
            ``'concat'``, this contains only leaf nodes. For ``'individual'``,
            this contains all cached completions/responses.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        if reward_discount is not None and not self._apply_reward_discount_called:
            self.apply_reward_discount(turn_discount=reward_discount)

        cache = self
        if len(cache) == 0:
            return {}

        # Filter out incomplete interactions (those still being processed)
        # This can happen when using anthropic agent sdk
        # where Claude Code CLI may send internal requests (e.g., git history analysis)
        # that are still in-flight when the main user request completes.
        complete_cache = {}
        for id, interaction in self.items():
            if (
                interaction.interaction_id is None
                or interaction.output_message_list is None
            ):
                logger.warning(
                    f"Skipping incomplete interaction during export: cache_key={id}, "
                    f"messages={interaction.messages[:1] if interaction.messages else []}..."
                )
                continue
            if interaction.interaction_id != id:
                raise ValueError(
                    f"Interaction ID mismatch: {interaction.interaction_id} != {id}"
                )
            complete_cache[id] = interaction

        if len(complete_cache) == 0:
            return {}

        if style == "concat":
            for interaction in complete_cache.values():
                if interaction.chat_template_type != "concat":
                    raise ValueError(
                        "Cannot export interactions in 'concat' style when "
                        "interaction.chat_template_type != 'concat' for any interaction. "
                        "This is because when applying chat template using some "
                        "tokenizers, there might be some tokens added or removed "
                        "(e.g. think tokens), making it impossible to construct the conversation tree. "
                        "Please use 'individual' style instead."
                    )

            # Build children mapping to find leaf nodes.
            has_children = set()
            for obj in complete_cache.values():
                if obj.parent is not None:
                    has_children.add(obj.parent.interaction_id)

            # Return only leaf nodes (nodes without children)
            return {
                id: interaction
                for id, interaction in complete_cache.items()
                if id not in has_children
            }
        elif style == "individual":
            return dict(**complete_cache)
        else:
            raise ValueError(f"Invalid export interactions style {style}")
