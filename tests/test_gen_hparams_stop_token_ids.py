# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api import ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import tokenizer_stop_token_ids

_DEFAULT_PAD_TOKEN_ID = 0
_DEFAULT_EOS_TOKEN_IDS = (128001, 128009)
_DEFAULT_STOP_TOKEN_IDS = [_DEFAULT_PAD_TOKEN_ID, *_DEFAULT_EOS_TOKEN_IDS]


class _TokenizerStub:
    def __init__(self, pad_token_id, eos_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [11, 12]}

    def decode(self, token_ids):
        return ""


def _stop_tokenizer(eos_token_id=None):
    if eos_token_id is None:
        eos_token_id = list(_DEFAULT_EOS_TOKEN_IDS)
    return _TokenizerStub(pad_token_id=_DEFAULT_PAD_TOKEN_ID, eos_token_id=eos_token_id)


class _EngineStub:
    def __init__(self):
        self.requests: list[ModelRequest] = []

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=[42, req.gconfig.stop_token_ids[0]],
            output_logprobs=[0.0],
            output_versions=[0],
            stop_reason="stop",
            tokenizer=req.tokenizer,
        )


@pytest.fixture
def stop_token_client():
    tokenizer = _stop_tokenizer()
    engine = _EngineStub()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        api_key="test",
        base_url="http://localhost",
    )
    return client, engine


class TestTokenizerStopTokenIds:
    @pytest.mark.parametrize(
        "pad_token_id, eos_token_id, expected",
        [(None, 2, [2]), (0, None, [0]), (0, [None, 2], [0, 2])],
    )
    def test_none_token_id_not_injected(self, pad_token_id, eos_token_id, expected):
        tokenizer = _TokenizerStub(pad_token_id=pad_token_id, eos_token_id=eos_token_id)

        new_gconfig = GenerationHyperparameters().new_with_stop_token_ids(
            tokenizer_stop_token_ids(tokenizer)
        )

        assert None not in new_gconfig.stop_token_ids
        assert new_gconfig.stop_token_ids == expected
        assert all(isinstance(tid, int) for tid in new_gconfig.stop_token_ids)

    def test_valid_ids_added_without_duplicates(self):
        gconfig = GenerationHyperparameters(stop_token_ids=[7])

        new_gconfig = gconfig.new_with_stop_token_ids([2, 2])

        assert new_gconfig.stop_token_ids == [7, 2]

    def test_overlapping_pad_and_eos_token_ids_deduplicated(self):
        tokenizer = _TokenizerStub(pad_token_id=2, eos_token_id=[2, 3])

        new_gconfig = GenerationHyperparameters().new_with_stop_token_ids(
            tokenizer_stop_token_ids(tokenizer)
        )

        assert new_gconfig.stop_token_ids == [2, 3]

    @pytest.mark.parametrize(
        "eos_token_id", [list(_DEFAULT_EOS_TOKEN_IDS), _DEFAULT_EOS_TOKEN_IDS]
    )
    def test_sequence_valued_eos_token_id(self, eos_token_id):
        tokenizer = _stop_tokenizer(eos_token_id=eos_token_id)
        gconfig = GenerationHyperparameters()

        new_gconfig = gconfig.new_with_stop_token_ids(
            tokenizer_stop_token_ids(tokenizer)
        )

        assert new_gconfig.stop_token_ids == _DEFAULT_STOP_TOKEN_IDS
        assert all(isinstance(tid, int) for tid in new_gconfig.stop_token_ids)

    def test_tokenizer_wrapper_matches_composed_helpers(self):
        tokenizer = _stop_tokenizer()
        gconfig = GenerationHyperparameters(stop_token_ids=[7])

        wrapped_gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        composed_gconfig = gconfig.new_with_stop_token_ids(
            tokenizer_stop_token_ids(tokenizer)
        )

        assert wrapped_gconfig.stop_token_ids == composed_gconfig.stop_token_ids

    def test_response_strips_list_valued_eos_token_id(self):
        tokenizer = _stop_tokenizer()
        response = ModelResponse(
            output_tokens=[42, _DEFAULT_EOS_TOKEN_IDS[-1]],
            stop_reason="stop",
            tokenizer=tokenizer,
        )

        assert response.end_with_stop
        assert response.output_tokens_without_stop == [42]


@pytest.mark.asyncio
async def test_chat_completions_flattens_list_valued_eos_token_id(stop_token_client):
    client, engine = stop_token_client
    await client.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=1,
    )

    stop_token_ids = engine.requests[0].gconfig.stop_token_ids
    assert stop_token_ids == _DEFAULT_STOP_TOKEN_IDS


@pytest.mark.asyncio
async def test_responses_flattens_list_valued_eos_token_id(stop_token_client):
    client, engine = stop_token_client
    await client.responses.create(input="hello", max_output_tokens=1, tools=[])

    stop_token_ids = engine.requests[0].gconfig.stop_token_ids
    assert stop_token_ids == _DEFAULT_STOP_TOKEN_IDS
