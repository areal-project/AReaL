# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AddedToken, PreTrainedTokenizerFast

from areal.api import ModelRequest
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.vision_tokens import protect_vision_tokens
from areal.utils.hf_utils import apply_chat_template
from areal.utils.sglang_image import IMAGE_SPANS_KEY, image_prompt_metadata
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


@pytest.fixture
def tokenizer():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.add_tokens(
        [
            AddedToken(token, normalized=False, special=True)
            for token in (
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            )
        ],
        special_tokens=True,
    )
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{% if m.content is string %}{{ m.content }}{% else %}"
        "{% for p in m.content %}{% if p.type == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}"
        "{% endfor %}{{ tools | tojson if tools else '' }}"
    )
    return tokenizer


@pytest.mark.parametrize(
    "literal",
    ["<|image_pad|>", "<|video_pad|>", "<|vision_start|><|image_pad|><|vision_end|>"],
)
def test_protect_visual_text_restores_ids_without_mutating_shared_tokenizer(
    tokenizer, literal
):
    messages = [
        {"role": "assistant", "content": f"left{literal}right"},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": literal}],
        },
    ]
    tools = [{"description": literal, literal: "tool"}]
    original = deepcopy((messages, tools))
    vocab = tokenizer.get_vocab()
    prompt = apply_chat_template(
        tokenizer, messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    local, protected, restore = protect_vision_tokens(
        processor, tokenizer, messages, tools, {}, prompt
    )
    # Only the typed image remains visible to the image processor.
    assert protected.count("<|image_pad|>") == 1
    actual = [restore.get(token, token) for token in local.tokenizer.encode(protected)]
    assert actual == tokenizer.encode(prompt)
    assert tokenizer.get_vocab() == vocab
    assert processor.tokenizer is tokenizer
    assert (messages, tools) == original


def test_clean_visual_prompt_keeps_shared_processor_and_tokenizer(tokenizer):
    processor = SimpleNamespace(tokenizer=tokenizer)
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = apply_chat_template(tokenizer, messages, tokenize=False)
    actual, text, restore = protect_vision_tokens(
        processor, tokenizer, messages, None, {}, prompt
    )
    assert actual is processor
    assert text == prompt
    assert restore == {}


def test_custom_template_keeps_real_image_placeholders(tokenizer):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "<|image_pad|>"}],
        }
    ]
    kwargs = {"chat_template": tokenizer.chat_template}
    prompt = apply_chat_template(tokenizer, messages, tokenize=False, **kwargs)
    local, text, restore = protect_vision_tokens(
        SimpleNamespace(tokenizer=tokenizer), tokenizer, messages, None, kwargs, prompt
    )
    assert text.count("<|image_pad|>") == 1
    assert [
        restore.get(token, token) for token in local.tokenizer.encode(text)
    ] == tokenizer.encode(prompt)


def test_unregistered_visual_text_is_ordinary_text(tokenizer):
    plain = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    processor = SimpleNamespace(tokenizer=plain)
    messages = [{"role": "user", "content": "<|image_pad|>"}]
    local, text, restore = protect_vision_tokens(
        processor, plain, messages, None, {}, "<|image_pad|>"
    )
    assert local is processor
    assert text == "<|image_pad|>"
    assert restore == {}


def test_none_metadata_is_accepted_without_images():
    assert image_prompt_metadata(None, [1, 2], None, None, None) == {}


def test_ambiguous_manual_qwen_request_requires_image_provenance(tokenizer):
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = SimpleNamespace(tokenizer=tokenizer, image_token_id=image_id)
    req = ModelRequest(
        input_ids=[3, image_id, image_id, 4, image_id],
        image_data=["jpeg"],
        processor=processor,
    )
    with pytest.raises(ValueError, match="explicit real-image spans"):
        SGLangBackend().build_generation_request(req, False, 0)


def test_text_image_pad_adjacent_to_real_span_is_not_a_delimiter(tokenizer):
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = SimpleNamespace(tokenizer=tokenizer, image_token_id=image_id)
    ids = [image_id] * 5
    req = ModelRequest(
        input_ids=ids,
        image_data=["jpeg"],
        processor=processor,
        metadata={IMAGE_SPANS_KEY: [[0, 3]]},
    )
    payload = SGLangBackend().build_generation_request(req, False, 0).payload
    assert payload["input_ids"] == ids
    assert payload["mm_sampling_kwargs"][IMAGE_SPANS_KEY] == [[0, 3]]


@pytest.mark.parametrize("backend_cls", [SGLangBackend, SGLangBridgeBackend])
def test_literal_visual_ids_use_explicit_offsets_with_unchanged_context(
    tokenizer, backend_cls
):
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = SimpleNamespace(tokenizer=tokenizer, image_token_id=image_id)
    ids = (
        tokenizer.encode("<|vision_start|>")
        + [image_id] * 4
        + tokenizer.encode("<|vision_end|>")
        + [image_id]
    )
    types = [0, 1, 1, 1, 1, 0, 0]
    metadata = image_prompt_metadata({"keep": True}, ids, types, ["jpeg"], processor)
    req = ModelRequest(
        input_ids=ids, image_data=["jpeg"], processor=processor, metadata=metadata
    )
    payload = backend_cls().build_generation_request(req, False, 0).payload
    assert payload["input_ids"] == ids
    assert payload["mm_sampling_kwargs"] == {IMAGE_SPANS_KEY: [[1, 4]]}
    assert req.input_ids == ids
    assert req.metadata["keep"]


def test_retry_switches_to_explicit_offsets_after_sampled_visual_token(tokenizer):
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids = (
        tokenizer.encode("<|vision_start|>")
        + [image_id] * 4
        + tokenizer.encode("<|vision_end|>")
    )
    processor = SimpleNamespace(tokenizer=tokenizer, image_token_id=image_id)
    metadata = image_prompt_metadata({}, ids, [0, 1, 1, 1, 1, 0], ["jpeg"], processor)
    req = ModelRequest(
        input_ids=ids, image_data=["jpeg"], processor=processor, metadata=metadata
    )
    backend = SGLangBridgeBackend()
    request = backend.build_generation_request(req, False, 0)
    assert len(request.payload["input_ids"]) == 3
    assert "mm_sampling_kwargs" not in request.payload
    backend.patch_generation_request(request, req, [image_id, image_id], 12)
    assert request.payload["input_ids"] == ids + [image_id, image_id]
    assert request.payload["mm_sampling_kwargs"][IMAGE_SPANS_KEY] == [[1, 4]]
    assert request.payload["sampling_params"]["max_new_tokens"] == 12


@pytest.mark.parametrize("spans", [[[1, 9]], [[1, 4], [3, 5]], [[0, 1]]])
def test_invalid_real_image_spans_fail_before_inference(tokenizer, spans):
    processor = SimpleNamespace(tokenizer=tokenizer, image_token_id=1)
    req = ModelRequest(
        input_ids=[3, 1, 1, 1, 1, 4],
        image_data=["jpeg"],
        processor=processor,
        metadata={IMAGE_SPANS_KEY: spans},
    )
    with pytest.raises(ValueError, match="span"):
        SGLangBackend().build_generation_request(req, False, 0)
