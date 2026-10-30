# SPDX-License-Identifier: Apache-2.0

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from areal.api import ModelResponse
from areal.experimental.openai.client import ArealOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["hf", "concat"])
@pytest.mark.parametrize("structured_input", [False, True])
async def test_responses_output_roundtrip_links_parent_and_preserves_actions(
    template, structured_input
):
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "user": 2, "assistant": 3, "task": 4, "answer": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    requests = []
    # Two sampled EOS tokens cannot be reconstructed from the visible output.
    sampled = [5, 1, 1]

    class Engine:
        async def agenerate(self, request):
            requests.append(request)
            return ModelResponse(
                input_tokens=list(request.input_ids),
                output_tokens=sampled.copy(),
                output_logprobs=[-0.2] * len(sampled),
                output_versions=[7] * len(sampled),
                stop_reason="stop",
                tokenizer=tokenizer,
            )

    content = [{"type": "input_text", "text": "task"}] if structured_input else "task"
    message = {"role": "user", "content": content}
    client = ArealOpenAI(
        engine=Engine(),
        tokenizer=tokenizer,
        chat_template_type=template,
        reasoning_parser="",
        api_key="test",
        base_url="http://test.invalid/v1",
    )
    try:
        first = await client.responses.create(input=[message], max_output_tokens=8)
        second = await client.responses.create(
            input=[message, *first.output, message], max_output_tokens=8
        )
        parent = client.get_interaction(first.id)
        child = client.get_interaction(second.id)
        assert child.parent is parent
        assert first.output[0].type == "message"
        assert parent.output_message_list == [
            {"role": "assistant", "content": "answer"}
        ]
        assert parent.model_response.output_tokens == sampled
        if template == "concat":
            prefix = requests[0].input_ids + sampled
            assert requests[1].input_ids[: len(prefix)] == prefix
            tensors = child.to_tensor_dict()
            start = len(requests[0].input_ids)
            assert tensors["logprobs"][
                0, start : start + len(sampled)
            ].tolist() == pytest.approx([-0.2] * len(sampled))
    finally:
        await client.close()
