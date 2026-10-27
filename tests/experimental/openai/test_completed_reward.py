"""Terminal rewards must survive export when an unfinished request trails a turn."""

from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from fastapi import HTTPException

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import client_session
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData, SetRewardRequest
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def interaction(cid, *, complete=True, messages=None):
    return InteractionWithTokenLogpReward(
        _interaction_id=cid if complete else None,
        messages=messages or [{"role": "user", "content": cid}],
        output_message_list=[{"role": "assistant", "content": cid}]
        if complete
        else None,
        chat_template_type="concat",
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_set_last_reward_unfinished_tail_preserves_reward_after_export(duplicate):
    a = interaction("a")
    b = interaction("b", complete=False, messages=a.messages if duplicate else None)
    cache = InteractionCache.from_dict({"a": a, "b": b})

    cache.set_last_reward(0.894)
    exported = cache.export_interactions(style="concat", drop_retry_orphans=True)

    assert list(exported) == ["a"]
    assert exported["a"].reward == pytest.approx(0.894)
    assert b.reward is None
    assert cache.total_reward == pytest.approx(0.894)


def test_set_last_reward_multiple_unfinished_entries_selects_latest_complete():
    a, b = interaction("a"), interaction("b")
    c, d = interaction("c", complete=False), interaction("d", complete=False)
    cache = InteractionCache.from_dict({"a": a, "b": b, "c": c, "d": d})

    cache.set_last_reward(0.8)
    cache.set_last_reward(0.9)

    assert [a.reward, b.reward, c.reward, d.reward] == [None, 0.9, None, None]
    assert cache.total_reward == pytest.approx(0.9)


@pytest.mark.parametrize("empty", [False, True])
def test_set_last_reward_no_completed_interaction_raises_without_assigning(empty):
    cache = InteractionCache.from_dict(
        {} if empty else {"a": interaction("a", complete=False)}
    )

    with pytest.raises(ValueError, match="No completed interactions"):
        cache.set_last_reward(0.9)

    assert cache.total_reward == 0
    assert all(item.reward is None for item in cache.values())


@pytest.mark.parametrize("explicit_id", [None, "b", "alias-b"])
def test_proxy_set_reward_unfinished_tail_preserves_explicit_id_semantics(
    monkeypatch, explicit_id
):
    session = SessionData(session_id="terminal-reward-test")
    a, b = interaction("a"), interaction("b", complete=False)
    session._completions = InteractionCache.from_dict({"a": a, "b": b})
    session.stream_completion_aliases["alias-b"] = "b"
    monkeypatch.setattr(srv, "_session_cache", {session.session_id: session})

    srv.set_reward(
        SetRewardRequest(interaction_id=explicit_id, reward=0.9), session.session_id
    )

    assert (a.reward, b.reward) == ((0.9, None) if explicit_id is None else (None, 0.9))


def test_proxy_set_last_reward_only_unfinished_entries_returns_error(monkeypatch):
    session = SessionData(session_id="terminal-reward-test")
    session._completions = InteractionCache.from_dict(
        {"a": interaction("a", complete=False)}
    )
    monkeypatch.setattr(srv, "_session_cache", {session.session_id: session})

    with pytest.raises(HTTPException, match="No completed interactions") as exc:
        srv.set_reward(SetRewardRequest(reward=0.9), session.session_id)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_set_last_reward_server_rejects_assignment_propagates_error(monkeypatch):
    error = aiohttp.ClientResponseError(
        Mock(), (), status=400, message="No completed interactions"
    )
    monkeypatch.setattr(
        client_session, "post_json_with_retry", AsyncMock(side_effect=error)
    )

    with pytest.raises(aiohttp.ClientResponseError):
        await client_session.set_last_interaction_reward(Mock(), 0.9)
