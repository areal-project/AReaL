# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import httpx
import pytest

from examples.weftmoe import dapo_math_sampling_mask as entrypoint


def test_baseline_reward_passes_gold_before_prediction(monkeypatch):
    """Preserve the old run's verifier argument order."""
    calls = []
    monkeypatch.setattr(entrypoint, "parse", lambda text: [text])
    monkeypatch.setattr(
        entrypoint,
        "verify",
        lambda gold, prediction: calls.append((gold, prediction)) or True,
    )
    assert entrypoint.reward_fn("prediction", "gold") == 1.0
    assert calls == [(["gold"], ["prediction"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("sampling_mask_transfer", [False, True])
async def test_both_arms_send_top_k_in_openai_request(
    monkeypatch, sampling_mask_transfer
):
    """Exercise the entrypoint, MathAgent and real SDK request serialization."""
    config = SimpleNamespace(
        sampling_mask_transfer=sampling_mask_transfer,
        tokenizer_path="unused-model",
        train_dataset=object(),
        gconfig=SimpleNamespace(
            temperature=1.0, top_p=0.95, top_k=30, max_new_tokens=8192
        ),
    )
    captured = {}

    class Trainer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(entrypoint, "load_expr_config", lambda *args: (config, []))
    monkeypatch.setattr(entrypoint, "load_hf_tokenizer", lambda *args: object())
    monkeypatch.setattr(entrypoint, "get_custom_dataset", lambda **kwargs: [])
    monkeypatch.setattr(entrypoint, "PPOTrainer", Trainer)
    entrypoint.main()

    async def reward(**kwargs):
        assert kwargs == {"completions": "prediction", "answer": "gold"}
        return 1.0

    def wrapper(fn):
        assert fn is entrypoint.reward_fn
        return reward

    monkeypatch.setattr(entrypoint, "AsyncRewardWrapper", wrapper)
    agent = entrypoint.BaselineMathAgent(**captured["workflow_kwargs"])

    def respond(request):
        body = json.loads(request.content)
        assert body["top_k"] == 30
        assert body["top_p"] == 0.95
        assert body["temperature"] == 1.0
        assert body["max_completion_tokens"] == 8192
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "prediction"},
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert (
            await agent.run(
                {"messages": "prompt", "answer": "gold"},
                http_client=client,
                base_url="http://localhost/v1",
                api_key="test-only",
            )
            == 1.0
        )
