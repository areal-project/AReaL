# SPDX-License-Identifier: Apache-2.0

"""Group processor sharing through v2 chat, export, and RTensor localization."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
import torch
from PIL import Image

from areal.api.cli_args import PRMConfig, PRMScorerConfig
from areal.infra.rpc import rtensor
from areal.infra.rpc.serialization import deserialize_value
from areal.v2.inference_service.data_proxy.app import (
    _create_areal_client,
    _create_inf_bridge,
    create_app,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import SessionStore


@pytest_asyncio.fixture
async def proxy(monkeypatch, request):
    """Use real cache/export code with fake inference and tensor transport."""
    config = DataProxyConfig(tokenizer_path="mock-vlm", backend_type="sglang")
    prm_enabled = getattr(request, "param", False)
    if prm_enabled:
        config = DataProxyConfig(
            tokenizer_path="mock-vlm",
            backend_type="sglang",
            chat_template_type="concat",
            prm=PRMConfig(
                scorers=[
                    PRMScorerConfig(
                        path="examples.prm.scorers.LengthBudgetScorer",
                        weight=0.5,
                        kwargs={"max_output_tokens": 32},
                    )
                ]
            ),
        )
    tokenizer = MagicMock(eos_token_id=99, pad_token_id=0)
    tokenizer.apply_chat_template.side_effect = lambda *args, tokenize=True, **kwargs: (
        {"input_ids": [10, 2, 20]} if tokenize else "describe"
    )
    tokenizer.decode.return_value = "answer"

    def process(**kwargs):
        # Fresh tensors on every real processor call: a missed cache must not
        # accidentally pass the alias checks because the mock reused an object.
        return {
            "input_ids": torch.tensor([[10, 2, 20]], dtype=torch.long),
            "pixel_values": torch.ones((4, 3), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }

    processor = MagicMock(image_processor=MagicMock(), side_effect=process)
    tok = SimpleNamespace(_tok=tokenizer, processor=processor)
    pause = PauseState()
    bridge = _create_inf_bridge("http://test.invalid", pause, config)
    send = AsyncMock(
        return_value={
            "meta_info": {
                "finish_reason": {"type": "length"},
                "output_token_logprobs": [(-0.1, 77)],
            }
        }
    )
    monkeypatch.setattr(bridge, "_send_request", send)
    areal_client = _create_areal_client(bridge, tok, config)
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.TokenizerProxy", lambda path: tok
    )
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app._create_inf_bridge",
        lambda *args: bridge,
    )
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app._create_areal_client",
        lambda *args: areal_client,
    )
    app = create_app(config)

    tensors = {}

    def save(tensor):
        key = str(len(tensors))
        tensors[key] = tensor
        return key

    backend = MagicMock()
    backend.store.side_effect = save
    backend.fetch.side_effect = lambda shards: [
        tensors[shard.shard_id].clone() for shard in shards
    ]
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    monkeypatch.setattr(rtensor, "_fetch_buffer", {})
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        try:
            yield SimpleNamespace(
                client=client,
                store=app.state.session_store,
                processor=processor,
                send=send,
                tensors=tensors,
                backend=backend,
                admin={"Authorization": f"Bearer {config.admin_api_key}"},
                prm_enabled=prm_enabled,
            )
        finally:
            await areal_client.close()


async def _run_group(proxy, colors):
    start = await proxy.client.post(
        "/rl/start_session",
        headers=proxy.admin,
        json={"task_id": "same-task", "group_size": len(colors)},
    )
    assert start.status_code == 201
    sessions = start.json()["sessions"]

    async def run(member, color, reward):
        content = "describe"
        if color is not None:
            with BytesIO() as buffer:
                Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content = [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        headers = {"Authorization": f"Bearer {member['session_api_key']}"}
        chat = await proxy.client.post(
            "/chat/completions",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": 1,
                # A request cannot select another session's processor cache.
                "processor_cache": "untrusted-cache",
            },
        )
        assert chat.status_code == 200, chat.text
        response = await proxy.client.post(
            "/rl/set_reward", headers=headers, json={"reward": reward}
        )
        assert response.status_code == 200, response.text

    await asyncio.gather(
        *(
            run(member, color, i % 2)
            for i, (member, color) in enumerate(zip(sessions, colors))
        )
    )
    return sessions


async def _export(proxy, sessions, *, discard=False):
    response = await proxy.client.post(
        "/export_trajectories",
        headers=proxy.admin,
        json={
            "session_ids": [member["session_id"] for member in sessions],
            "style": "concat" if proxy.prm_enabled else "individual",
            "discard_trajectory": discard,
        },
    )
    assert response.status_code == 200, response.text
    return deserialize_value(response.json()["traj"])


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy", [False, True], indirect=True)
@pytest.mark.parametrize(
    "colors,unique_images", [(["red"] * 4, 1), (["red", "blue"], 2), (["red"], 1)]
)
async def test_group_export_shares_only_matching_images(proxy, colors, unique_images):
    """One processor call/shard/fetch per shared image, with all samples retained."""
    sessions = await _run_group(proxy, colors)
    assert proxy.processor.call_count == unique_images
    assert proxy.send.await_count == len(colors)
    group_cache = proxy.store.get_session(sessions[0]["session_id"]).processor_cache
    trajectory = await _export(proxy, sessions)
    assert proxy.store.session_count == 0
    if group_cache is not None:
        assert not group_cache._results
        assert group_cache._closed

    mm = trajectory["multi_modal_input"]
    shard_ids = {item["pixel_values"].shard.shard_id for item in mm}
    assert len(mm) == len(colors)
    assert len(shard_ids) == unique_images
    assert (
        sum(proxy.tensors[sid].nbytes for sid in shard_ids) == unique_images * 4 * 3 * 4
    )

    # JSON decoding creates distinct RTensor wrappers. Localization must still
    # fetch each shard once and restore the aliases before training consumes it.
    local = rtensor.RTensor.localize(trajectory, preserve_tensor_aliases=True)
    fetched = [shard.shard_id for shard in proxy.backend.fetch.call_args.args[0]]
    assert len(fetched) == len(set(fetched))
    for sid in shard_ids:
        assert fetched.count(sid) == 1
    images = [item["pixel_values"] for item in local["multi_modal_input"]]
    assert len({id(image) for image in images}) == unique_images
    assert torch.cat(images).shape == torch.Size([len(colors) * 4, 3])
    torch.testing.assert_close(
        local["input_ids"],
        torch.tensor([[10, 2, 20, 77]] * len(colors)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        local["rewards"],
        torch.tensor([float(i % 2) for i in range(len(colors))]),
        rtol=0,
        atol=0,
    )
    if proxy.prm_enabled:
        torch.testing.assert_close(
            local["token_rewards"],
            torch.tensor([[0.0, 0.0, 0.0, 0.5]] * len(colors)),
            rtol=0,
            atol=0,
        )
        assert local["mm_token_type_ids"].shape == local["input_ids"].shape
        assert not local["mm_token_type_ids"][:, -1].any()
    else:
        assert "token_rewards" not in local


@pytest.mark.asyncio
async def test_group_cache_isolated_across_groups_and_discarded_on_failure(proxy):
    """Identical task IDs do not share across groups; discard creates no shards."""
    first = await _run_group(proxy, ["red", "red"])
    second = await _run_group(proxy, ["red", "red"])
    assert proxy.processor.call_count == 2
    first_cache = proxy.store.get_session(first[0]["session_id"]).processor_cache
    second_cache = proxy.store.get_session(second[0]["session_id"]).processor_cache
    assert first_cache is not second_cache
    assert await _export(proxy, first, discard=True) == {}
    assert not proxy.tensors
    assert first_cache._closed
    assert not second_cache._closed
    await _export(proxy, second)
    assert second_cache._closed


@pytest.mark.asyncio
async def test_discard_unrewarded_group_removes_sessions_and_cache(proxy):
    """A failed agent needs no fallback reward before discard can clean up."""
    response = await proxy.client.post(
        "/rl/start_session",
        headers=proxy.admin,
        json={"task_id": "failed-task", "group_size": 2},
    )
    assert response.status_code == 201
    sessions = response.json()["sessions"]
    cache = proxy.store.get_session(sessions[0]["session_id"]).processor_cache
    cache.get_or_compute("image", lambda: torch.ones((2, 3)))

    assert await _export(proxy, sessions, discard=True) == {}

    assert cache._closed
    assert not cache._results
    for member in sessions:
        assert proxy.store.get_session(member["session_id"]) is None
        assert proxy.store.get_session_by_api_key(member["session_api_key"]) is None
    proxy.backend.store.assert_not_called()


@pytest.mark.asyncio
async def test_text_group_does_not_call_processor_or_add_image_tensors(proxy):
    """Text requests retain independent samples and bypass the processor."""
    trajectory = await _export(proxy, await _run_group(proxy, [None, None]))
    proxy.processor.assert_not_called()
    # A loaded processor adds empty multimodal metadata even for text-only
    # samples; this must not contain image tensors or invoke the processor.
    assert trajectory["multi_modal_input"] == [{}, {}]
    assert trajectory["input_ids"].shape == torch.Size([2, 4])
    token_types = rtensor.RTensor.localize(trajectory["mm_token_type_ids"])
    torch.testing.assert_close(
        token_types, torch.zeros((2, 4), dtype=torch.long), rtol=0, atol=0
    )


@pytest.mark.parametrize("cleanup", ["remove", "stale"])
def test_session_cleanup_releases_cache_only_after_last_member(cleanup):
    """A sibling remains usable after one lease ends, including stale cleanup."""
    store = SessionStore()
    first, _ = store.start_session(
        "task", processor_cache_group_id="group", processor_cache_group_size=2
    )
    second, _ = store.start_session(
        "task", processor_cache_group_id="group", processor_cache_group_size=2
    )
    cache = store.get_session(first).processor_cache
    assert cache is store.get_session(second).processor_cache
    token = object()
    assert cache.get_or_compute("key", lambda: token) is token
    if cleanup == "remove":
        store.remove_session(first)
        store.remove_session(first)  # Removing twice must not release the sibling.
    else:
        store.get_session(first)._last_access_time = 0
        store.cleanup_stale()
    assert cache.get_or_compute("key", lambda: object()) is token
    store.remove_session(second)
    assert cache._closed
    assert not cache._results
