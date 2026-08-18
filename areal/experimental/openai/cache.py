# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

logger = logging.getLogger("OpenAICache")

# Debug-only: dumping mismatched parent/child messages is OFF by default because
# the payloads can contain full conversations. Set the dump dir env var to opt in.
_MISMATCH_DUMP_DIR_ENV = "AREAL_OPENAI_CACHE_MISMATCH_DUMP_DIR"
_MISMATCH_DUMP_LIMIT_ENV = "AREAL_OPENAI_CACHE_MISMATCH_DUMP_LIMIT"
_DEFAULT_MISMATCH_DUMP_LIMIT = 20


@runtime_checkable
class PrefixMatcher(Protocol):
    """Protocol for custom message prefix matching functions."""

    def __call__(self, a: list[dict], b: list[dict]) -> bool: ...


def default_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    """Default exact prefix matcher: ``b[:len(a)] == a``."""
    if len(a) > len(b):
        return False
    return b[: len(a)] == a


class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    def __init__(
        self,
        *args,
        session_id: str = "unknown",
        prefix_matcher: PrefixMatcher
        | Callable[[list[dict], list[dict]], bool]
        | None = None,
        **kwargs,
    ):
        self._match_fail_count = 0
        self._session_id = session_id
        self._prefix_matcher: Callable[[list[dict], list[dict]], bool] = (
            prefix_matcher if prefix_matcher is not None else default_prefix_matcher
        )
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
        new = InteractionCache(
            session_id=self._session_id,
            prefix_matcher=self._prefix_matcher,
        )
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

    def _find_retry_orphan_ids(self) -> set[str]:
        """Identify retry-orphan completions/responses.

        Among a group of interactions sharing identical input messages
        (deep-equal), an entry is a retry orphan when it was not the one the
        agent actually consumed:

          1) If the group contains an entry with children (a later turn
             adopted it as parent), that entry is the consumed completion and
             every childless sibling is an orphan.
          2) If the whole group is childless (the session ended right after a
             retry, before any later turn could establish parentage), the tree
             cannot disambiguate. Fall back to generation time: keep the entry
             with the largest ``created_at`` (most likely the retry) and treat
             the earlier duplicates as orphans, so the consumed completion is
             not lost.

        This pattern is produced when the upstream Agent SDK times out
        waiting for a response, retries the same request, and the proxy
        records both the orphaned (never-delivered) completion and the
        retry. The retry's output is what the agent actually saw and
        continued from, while the orphan dangles as a leaf.
        """
        # Build has_children set from parent pointers wired up at __setitem__.
        has_children: set[str] = set()
        for entry in self.values():
            if entry.parent is not None:
                pid = entry.parent.interaction_id
                if pid is not None:
                    has_children.add(pid)

        # Group entries by hashed input messages, preserving insertion order.
        groups: dict[str, list[str]] = defaultdict(list)
        for cid, entry in self.items():
            if entry.messages is None:
                continue
            try:
                serialized = json.dumps(
                    entry.messages, sort_keys=True, default=str
                ).encode("utf-8")
            except (TypeError, ValueError):
                continue
            digest = hashlib.sha1(serialized).hexdigest()
            groups[digest].append(cid)

        orphan_ids: set[str] = set()
        for cids in groups.values():
            if len(cids) < 2:
                continue
            # Confirm the group is actually message-equal (hash collisions are
            # astronomically unlikely but the cost of confirming is trivial).
            ref_msgs = self[cids[0]].messages
            if any(self[c].messages != ref_msgs for c in cids[1:]):
                continue
            leaves = [c for c in cids if c not in has_children]
            if len(leaves) < len(cids):
                # At least one duplicate has a child: the tree tells us which
                # entry was actually consumed (a later turn adopted it as
                # parent), so every childless sibling is an orphan.
                orphan_ids.update(leaves)
            else:
                # The whole group is leaves. This happens when the session
                # ends right after a retry, before any later turn could adopt
                # the real completion as a parent. The tree can no longer
                # distinguish orphan from retry, so fall back to generation
                # time: keep the most recently created entry (most likely the
                # retry the agent consumed) and drop the earlier duplicates.
                # Tie-break on insertion order so a coarse (second-resolution)
                # or missing created_at still keeps the latest deterministically.
                order = {c: i for i, c in enumerate(cids)}
                survivor = max(
                    cids,
                    key=lambda c: (
                        self[c].created_at
                        if self[c].created_at is not None
                        else float("-inf"),
                        order[c],
                    ),
                )
                orphan_ids.update(c for c in cids if c != survivor)
        return orphan_ids

    def drop_retry_orphans(self) -> list[str]:
        """Remove retry-orphan entries in place; return the dropped IDs.

        Must be called BEFORE :meth:`apply_reward_discount`, otherwise the
        orphan rewards have already polluted the discounted chain.

        Acquires ``self._lock`` for the find-and-delete so it stays consistent
        with concurrent :meth:`set_reward` / insertion, and decrements
        ``self._total_reward`` by each dropped entry's reward to keep the
        running total accurate.
        """
        if self._apply_reward_discount_called:
            raise RuntimeError(
                "drop_retry_orphans must be called BEFORE apply_reward_discount."
            )
        with self._lock:
            orphan_ids = self._find_retry_orphan_ids()
            for oid in orphan_ids:
                self._total_reward -= self[oid].reward or 0.0
                del self[oid]
        if orphan_ids:
            logger.info(
                "Session %s: dropped %d retry-orphan completion(s): %s",
                self._session_id,
                len(orphan_ids),
                sorted(orphan_ids),
            )
        return list(orphan_ids)

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

        # Construct parent-child relationships using longest prefix rule.
        interactions = sorted(
            self.values(), key=lambda x: len(x.messages), reverse=True
        )
        child_msgs = value.messages

        for parent in interactions:
            # Skip interactions that are still being processed (output_message_list not set yet)
            # This can happen with concurrent requests where a streaming request hasn't
            # finished setting up yet. Such interactions cannot be parents anyway.
            if parent.output_message_list is None or parent.messages is None:
                continue
            parent_data = parent.messages + parent.output_message_list
            if self._prefix_matcher(parent_data, child_msgs):
                value.parent = parent
                break
            elif self._prefix_matcher(parent.messages, child_msgs):
                self._match_fail_count += 1

                logger.warning(
                    "Prefix mismatch (occurrence %d, session=%s): "
                    "parent.messages (len=%d) is a prefix of child (len=%d), "
                    "but parent_data (messages+output, len=%d) is not.",
                    self._match_fail_count,
                    self._session_id,
                    len(parent.messages),
                    len(child_msgs),
                    len(parent_data),
                )
                self._dump_mismatch(
                    parent_data=parent_data,
                    child_msgs=child_msgs,
                    parent_id=parent.interaction_id,
                    child_key=key,
                )

                is_similar, diff_a, diff_b = _is_similar_on_last_message(
                    parent_data, child_msgs
                )
                if is_similar:
                    logger.warning(
                        "Found a parent interaction with similar last message content, "
                        "but not a strict prefix match (occurrence %d, "
                        "session=%s, parent_len=%d, child_len=%d). "
                        "If you wish to use concat mode and build a conversation tree:\n"
                        "1. For completion, append `chat_completion.choices[0].message.model_dump()` to your messages.\n"
                        "2. For response, extend `[o.model_dump() for o in response.output]` to your messages.\n"
                        "Different keys in parent last message: %s\n"
                        "Different keys in child last message: %s",
                        self._match_fail_count,
                        self._session_id,
                        len(parent_data),
                        len(child_msgs),
                        diff_a,
                        diff_b,
                    )
        super().__setitem__(key, value)

    def _dump_mismatch(
        self,
        parent_data: list[dict],
        child_msgs: list[dict],
        parent_id: str | None,
        child_key: str,
    ) -> None:
        """Dump mismatched parent/child messages to a JSON file for debugging."""
        dump_dir_env = os.environ.get(_MISMATCH_DUMP_DIR_ENV)
        if not dump_dir_env:
            return
        try:
            dump_limit = int(
                os.environ.get(
                    _MISMATCH_DUMP_LIMIT_ENV, str(_DEFAULT_MISMATCH_DUMP_LIMIT)
                )
            )
            if dump_limit > 0 and self._match_fail_count > dump_limit:
                return

            dump_dir = Path(dump_dir_env)
            dump_dir.mkdir(parents=True, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"mismatch_{self._session_id}_{self._match_fail_count}_{ts}.json"
            dump_path = dump_dir / filename

            first_diff_idx = None
            for i, (pm, cm) in enumerate(zip(parent_data, child_msgs)):
                if pm != cm:
                    first_diff_idx = i
                    break

            dump = {
                "session_id": self._session_id,
                "occurrence": self._match_fail_count,
                "parent_id": parent_id,
                "child_key": child_key,
                "parent_len": len(parent_data),
                "child_len": len(child_msgs),
                "first_diff_idx": first_diff_idx,
                "parent_data": parent_data,
                "child_msgs": child_msgs,
            }

            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, ensure_ascii=False, default=str)

            logger.info("Mismatch dump saved to %s", dump_path)
        except Exception as e:
            logger.warning("Failed to dump mismatch: %s", e)

    def export_interactions(
        self,
        style: str,
        reward_discount: float | None = None,
        drop_retry_orphans: bool = False,
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
        if drop_retry_orphans:
            self.drop_retry_orphans()
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
