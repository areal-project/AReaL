# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from areal.experimental.openai import ArealOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat_completions", "responses"])
async def test_prompt_construction_failure_discards_cached_interaction(api):
    """A rejected prompt must not leave an incomplete interaction in the cache."""
    engine = AsyncMock()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=object(),
        api_key="test",
        chat_template_type="concat",
    )

    with (
        patch(
            "areal.experimental.openai.client.concat_prompt_token_ids_with_parent",
            side_effect=ValueError("invalid parent token prefix"),
        ),
        pytest.raises(ValueError, match="invalid parent token prefix"),
    ):
        if api == "chat_completions":
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "hello"}]
            )
        else:
            await client.responses.create(input="hello")

    assert not client._cache
    engine.agenerate.assert_not_awaited()
