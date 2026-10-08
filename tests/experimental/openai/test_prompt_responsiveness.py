"""Keep agent prompt preparation from blocking proxy control requests."""

import asyncio
import threading

import pytest

from areal.experimental.openai import client


@pytest.mark.asyncio
async def test_prepare_hf_prompt_yields_while_tokenizing(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_template(*args, **kwargs):
        started.set()
        assert release.wait(timeout=3)
        return [1, 2, 3]

    monkeypatch.setattr(client, "apply_chat_template", slow_template)
    task = asyncio.create_task(
        client._prepare_prompt(
            tokenizer=object(),
            processor=None,
            tokenizer_messages=[{"role": "user", "content": "hello"}],
            concat_messages=[],
            image_data=[],
            parent=None,
            chat_template_type="hf",
            tools=None,
            extra_body={},
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.5)
    finally:
        release.set()
    assert (await task).input_ids == [1, 2, 3]
