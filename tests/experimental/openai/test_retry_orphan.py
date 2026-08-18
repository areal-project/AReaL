"""Unit tests for retry-orphan detection in InteractionCache.

Covers the scenario where the upstream Agent SDK times out, retries the same
LLM request, and the proxy ends up with two completions sharing the same
input messages. Only the latter is actually consumed by the agent; the
former is an "orphan" leaf that should be dropped before export.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _make_interaction(
    cid: str,
    messages: list[dict],
    output_messages: list[dict] | None = None,
    reward: float | None = None,
    created: int = 0,
) -> InteractionWithTokenLogpReward:
    """Build a minimal InteractionWithTokenLogpReward usable by the cache.

    Avoids loading a real tokenizer / openai schema by patching the
    ``completion`` attribute with a MagicMock that exposes the ``id``
    needed by ``interaction_id``.
    """
    interaction = InteractionWithTokenLogpReward(
        messages=messages,
        output_message_list=output_messages
        if output_messages is not None
        else [{"role": "assistant", "content": f"out-{cid}"}],
        chat_template_type="concat",
        reward=reward,
    )
    completion = MagicMock()
    completion.id = cid
    completion.created = created
    interaction.completion = completion
    return interaction


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_no_retry_keeps_all():
    """No duplicate-input pairs → nothing dropped."""
    cache = InteractionCache()
    cache["a"] = _make_interaction("a", [_user_msg("hi")])
    cache["b"] = _make_interaction(
        "b",
        [_user_msg("hi"), _assistant_msg("out-a"), _user_msg("more")],
    )
    dropped = cache.drop_retry_orphans()
    assert dropped == []
    assert set(cache.keys()) == {"a", "b"}


def test_single_retry_with_subsequent_turn_drops_orphan():
    """orphan(leaf) + retry(has child) → drop orphan."""
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["orphan"] = _make_interaction("orphan", msgs)  # leaf
    cache["retry"] = _make_interaction("retry", msgs)  # leaf for now
    # Subsequent turn whose messages extend retry's output → retry becomes parent
    cache["next"] = _make_interaction(
        "next",
        msgs + [_assistant_msg("out-retry"), _user_msg("more")],
    )
    # Sanity: prefix-matcher should have made "next" a child of "retry"
    assert cache["next"].parent is cache["retry"]
    dropped = cache.drop_retry_orphans()
    assert dropped == ["orphan"]
    assert set(cache.keys()) == {"retry", "next"}


def test_retry_then_session_ends_keeps_latest():
    """Two siblings, both leaves → keep the latest, drop the earlier.

    When the session ends immediately after a retry there is no later turn
    to anchor the real completion as a parent, so the tree cannot tell orphan
    from retry. The fallback keeps the most recently created entry (here the
    later-inserted ``retry``) instead of dropping both, avoiding data loss.
    """
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["orphan"] = _make_interaction("orphan", msgs)
    cache["retry"] = _make_interaction("retry", msgs)
    dropped = cache.drop_retry_orphans()
    assert dropped == ["orphan"]
    assert "retry" in cache
    assert "orphan" not in cache


def test_all_leaves_keeps_latest_by_created_at():
    """All-leaf fallback ranks by created_at, not insertion order.

    The retry is inserted *before* the orphan but carries a larger
    ``created_at`` (it was generated later, after the SDK timeout). The
    survivor must be the retry, proving generation time drives the tie, not
    the order rows landed in the cache.
    """
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["retry"] = _make_interaction("retry", msgs, created=100)
    cache["orphan"] = _make_interaction("orphan", msgs, created=50)
    dropped = cache.drop_retry_orphans()
    assert dropped == ["orphan"]
    assert set(cache.keys()) == {"retry"}


def test_three_retries_drops_two_orphans():
    """Three same-input leaves → keep the latest, drop the two earlier."""
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["o1"] = _make_interaction("o1", msgs, created=1)
    cache["o2"] = _make_interaction("o2", msgs, created=2)
    cache["final"] = _make_interaction("final", msgs, created=3)
    dropped = set(cache.drop_retry_orphans())
    assert dropped == {"o1", "o2"}
    assert set(cache.keys()) == {"final"}


def test_mid_conversation_retry():
    """Retry happens at turn 3 of a 5-turn dialogue.

    Only the orphan completion at that position is dropped; earlier and
    later turns are kept intact.
    """
    cache = InteractionCache()
    m1 = [_user_msg("q1")]
    cache["c1"] = _make_interaction("c1", m1)

    m2 = m1 + [_assistant_msg("out-c1"), _user_msg("q2")]
    cache["c2"] = _make_interaction("c2", m2)

    m3 = m2 + [_assistant_msg("out-c2"), _user_msg("q3")]
    cache["c3_orphan"] = _make_interaction("c3_orphan", m3)
    cache["c3_retry"] = _make_interaction("c3_retry", m3)

    m4 = m3 + [_assistant_msg("out-c3_retry"), _user_msg("q4")]
    cache["c4"] = _make_interaction("c4", m4)

    m5 = m4 + [_assistant_msg("out-c4"), _user_msg("q5")]
    cache["c5"] = _make_interaction("c5", m5)

    dropped = cache.drop_retry_orphans()
    assert dropped == ["c3_orphan"]
    assert "c3_orphan" not in cache
    assert set(cache.keys()) == {"c1", "c2", "c3_retry", "c4", "c5"}


def test_both_siblings_have_children_keeps_both():
    """Conservative: if both same-input entries have children, keep both."""
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["a"] = _make_interaction("a", msgs)
    cache["b"] = _make_interaction("b", msgs)
    # Branch off "a"
    cache["a_child"] = _make_interaction(
        "a_child", msgs + [_assistant_msg("out-a"), _user_msg("ka")]
    )
    # Branch off "b"
    cache["b_child"] = _make_interaction(
        "b_child", msgs + [_assistant_msg("out-b"), _user_msg("kb")]
    )
    assert cache["a_child"].parent is cache["a"]
    assert cache["b_child"].parent is cache["b"]
    dropped = cache.drop_retry_orphans()
    assert dropped == []
    assert set(cache.keys()) == {"a", "b", "a_child", "b_child"}


def test_drop_must_precede_discount():
    cache = InteractionCache()
    cache["a"] = _make_interaction("a", [_user_msg("hi")], reward=1.0)
    cache.apply_reward_discount(turn_discount=1.0)
    with pytest.raises(
        RuntimeError,
        match="drop_retry_orphans must be called BEFORE apply_reward_discount",
    ):
        cache.drop_retry_orphans()


def test_export_with_drop_retry_orphans_flag():
    """End-to-end: export_interactions(drop_retry_orphans=True) removes orphan."""
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["orphan"] = _make_interaction("orphan", msgs, reward=0.0)
    cache["retry"] = _make_interaction("retry", msgs, reward=1.0)
    cache["next"] = _make_interaction(
        "next",
        msgs + [_assistant_msg("out-retry"), _user_msg("more")],
        reward=2.0,
    )
    exported = cache.export_interactions(
        style="individual",
        reward_discount=1.0,
        drop_retry_orphans=True,
    )
    assert "orphan" not in exported
    assert set(exported.keys()) == {"retry", "next"}
    # Orphan's reward must not have contaminated the discounted chain:
    # discounted = next.reward(2.0) propagates to retry → 2.0 + 1.0 = 3.0
    assert exported["next"].reward == pytest.approx(2.0)
    assert exported["retry"].reward == pytest.approx(3.0)


def test_drop_updates_total_reward():
    """Dropping an orphan decrements the running total_reward."""
    msgs = [_user_msg("hi")]
    cache = InteractionCache()
    cache["orphan"] = _make_interaction("orphan", msgs)
    cache["retry"] = _make_interaction("retry", msgs)
    cache["next"] = _make_interaction(
        "next", msgs + [_assistant_msg("out-retry"), _user_msg("more")]
    )
    # Rewards set via set_reward feed the running _total_reward.
    cache.set_reward("orphan", 5.0)
    cache.set_reward("retry", 3.0)
    assert cache.total_reward == pytest.approx(8.0)
    dropped = cache.drop_retry_orphans()
    assert dropped == ["orphan"]
    # The orphan's 5.0 must be removed from the running total, leaving retry's.
    assert cache.total_reward == pytest.approx(3.0)
