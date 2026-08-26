# SPDX-License-Identifier: Apache-2.0

"""Transport contract for the vLLM backend's generation requests.

Text and multimodal prompts both go to ``/inference/v1/generate``, which
consumes the token ids we send verbatim, so the prompt training scores is the
prompt that ran. Media travels beside those ids as ``content_parts`` rather
than as messages, so the server never re-renders the conversation.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.vllm_remote import VLLMBackend


def _image_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


IMAGE_PAD = 151655
VIDEO_PAD = 151656


class _QwenLikeTokenizer:
    """Resolves the Qwen media placeholders and nothing else."""

    _ids = {"<|image_pad|>": IMAGE_PAD, "<|video_pad|>": VIDEO_PAD}

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token)

    def convert_ids_to_tokens(self, token_id):
        for token, tid in self._ids.items():
            if tid == token_id:
                return token
        return None


def _vision_request(gconfig: GenerationHyperparameters) -> ModelRequest:
    """A one-image request carrying both prompt representations.

    ``input_ids`` is vision-expanded (4 placeholders); ``collapsed_input_ids``
    is the same prompt with the single unexpanded placeholder vLLM expects.
    A tokenizer is mandatory: the placeholder-count guard refuses to run blind.
    """
    return ModelRequest(
        input_ids=[11, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 12],
        collapsed_input_ids=[11, IMAGE_PAD, 12],
        gconfig=gconfig,
        image_data=[_image_b64()],
        tokenizer=_QwenLikeTokenizer(),
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def test_text_request_sends_token_ids_to_the_generate_endpoint():
    """Test that a text prompt is sent as token ids, not re-rendered text."""
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    req = ModelRequest(input_ids=[11, 12, 13], gconfig=gconfig)

    http_req = VLLMBackend().build_generation_request(req, with_lora=False, version=0)

    assert http_req.endpoint == "/inference/v1/generate"
    assert http_req.payload["token_ids"] == [11, 12, 13]
    # The prompt must be carried as ids only; nothing may re-render it.
    assert "prompt" not in http_req.payload
    assert "messages" not in http_req.payload


def test_text_request_nests_sampling_params():
    """Test that sampling options move under the native sampling_params object."""
    gconfig = GenerationHyperparameters(
        max_new_tokens=8, frequency_penalty=0.5, stop=["STOP"], top_p=0.9
    )
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    sampling_params = payload["sampling_params"]
    assert sampling_params["frequency_penalty"] == 0.5
    assert sampling_params["stop"] == ["STOP"]
    assert sampling_params["top_p"] == 0.9
    assert sampling_params["max_tokens"] == 8
    # Flat OpenAI-style keys must not leak alongside the nested object.
    assert "frequency_penalty" not in payload


def test_stop_strings_are_forwarded_with_detokenization_left_on():
    """Test that stop strings survive, which requires server-side detokenization.

    vLLM raises when `stop` is set and `detokenize` is False, so the request
    must never disable detokenization as an optimisation.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=4, stop=["STOP"])
    req = ModelRequest(input_ids=[11], gconfig=gconfig)

    sampling_params = (
        VLLMBackend()
        .build_generation_request(req, with_lora=False, version=0)
        .payload["sampling_params"]
    )

    assert sampling_params["stop"] == ["STOP"]
    assert "detokenize" not in sampling_params


def test_greedy_maps_to_zero_temperature():
    """Test that greedy decoding is expressed as temperature 0."""
    gconfig = GenerationHyperparameters(max_new_tokens=4, greedy=True, temperature=0.7)
    req = ModelRequest(input_ids=[11], gconfig=gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["sampling_params"]["temperature"] == 0.0


def test_beam_search_is_rejected_rather_than_silently_dropped():
    """Test that an unsupported option fails loudly.

    use_beam_search is not a native SamplingParams field, so forwarding it
    would silently change decoding behaviour.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=4, use_beam_search=True)
    req = ModelRequest(input_ids=[11], gconfig=gconfig)

    with pytest.raises(NotImplementedError, match="beam search"):
        VLLMBackend().build_generation_request(req, with_lora=False, version=0)


def test_lora_names_the_versioned_adapter_on_the_generate_endpoint():
    """Test that LoRA still selects an adapter under the new transport."""
    gconfig = GenerationHyperparameters(max_new_tokens=4, lora_name="adapter")
    req = ModelRequest(input_ids=[11], gconfig=gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=True, version=3).payload
    )

    assert "adapter" in payload["model"]


def test_lora_with_an_empty_name_fails():
    """Test that enabling LoRA with a blank adapter name is an error."""
    gconfig = GenerationHyperparameters(max_new_tokens=4, lora_name="")
    req = ModelRequest(input_ids=[11], gconfig=gconfig)

    with pytest.raises(ValueError, match="lora_name"):
        VLLMBackend().build_generation_request(req, with_lora=True, version=0)


def test_vision_request_uses_the_same_endpoint_with_raw_media():
    """Test that images ride along with exact token ids on one endpoint.

    No messages cross the inference boundary, so the server cannot re-render
    the prompt into something other than what training will score.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=8, stop=["STOP"])
    http_req = VLLMBackend().build_generation_request(
        _vision_request(gconfig), with_lora=False, version=0
    )

    assert http_req.endpoint == "/inference/v1/generate"
    assert "messages" not in http_req.payload
    assert http_req.payload["sampling_params"]["stop"] == ["STOP"]
    assert http_req.payload["content_parts"][0]["url"].startswith(
        "data:image/png;base64,"
    )


def test_vision_request_sends_collapsed_ids_and_expects_the_expanded_prompt():
    """Test that the wire carries the collapsed prompt and the expanded one.

    vLLM expands placeholders itself, so sending the expanded prompt would
    expand it twice; expected_token_ids is what the server must reproduce.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    req = _vision_request(gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["token_ids"] == req.collapsed_input_ids
    assert payload["expected_token_ids"] == req.input_ids
    assert payload["token_ids"] != payload["expected_token_ids"]


def test_vision_request_without_a_collapsed_prompt_fails():
    """Test that a multimodal request missing the collapsed form is an error.

    Falling back to input_ids would silently double-expand the placeholders.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    req = _vision_request(gconfig)
    req.collapsed_input_ids = None

    with pytest.raises(ValueError, match="collapsed_input_ids"):
        VLLMBackend().build_generation_request(req, with_lora=False, version=0)


def test_text_request_sends_no_media_or_expectation():
    """Test that the text path stays on the plain token-in branch."""
    req = ModelRequest(
        input_ids=[11, 12], gconfig=GenerationHyperparameters(max_new_tokens=4)
    )

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert "content_parts" not in payload
    assert "expected_token_ids" not in payload


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_reads_token_ids_directly_from_the_generate_response():
    """Test that the native response needs no "token:123" string parsing."""
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "token_ids": [5, 6, 7],
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.2},
                        {"token": "c", "logprob": -0.3},
                    ]
                },
            }
        ]
    }

    result = VLLMBackend().parse_generation_response(response)

    assert result.output_tokens == [5, 6, 7]
    assert result.output_logprobs == [-0.1, -0.2, -0.3]
    assert result.stop_reason == "stop"


def test_parse_rejects_desynced_tokens_and_logprobs():
    """Test that parallel sequences of unequal length are an error.

    Silently zipping them would misattribute every logprob after the gap.
    """
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "token_ids": [5, 6, 7],
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
            }
        ]
    }

    with pytest.raises(ValueError, match="parallel"):
        VLLMBackend().parse_generation_response(response)


def test_extend_prompt_advances_both_forms_together():
    """Test that resuming an interrupted generation keeps the forms aligned.

    Advancing only one would make the server's expansion disagree with
    expected_token_ids, failing every retry after a weight update.
    """
    req = ModelRequest(
        input_ids=[1, 2, 3],
        collapsed_input_ids=[1, 9, 3],
        gconfig=GenerationHyperparameters(max_new_tokens=4),
    )

    req.extend_prompt([77, 78])

    assert req.input_ids == [1, 2, 3, 77, 78]
    assert req.collapsed_input_ids == [1, 9, 3, 77, 78]


def test_extend_prompt_on_a_text_request_leaves_no_collapsed_form():
    """Test that text-only requests keep a single representation."""
    req = ModelRequest(
        input_ids=[1, 2], gconfig=GenerationHyperparameters(max_new_tokens=4)
    )

    req.extend_prompt([5])

    assert req.input_ids == [1, 2, 5]
    assert req.collapsed_input_ids is None


def test_copy_preserves_the_collapsed_form_independently():
    """Test that copies do not share the collapsed prompt list."""
    req = ModelRequest(
        input_ids=[1, 2],
        collapsed_input_ids=[1, 9],
        gconfig=GenerationHyperparameters(max_new_tokens=4),
    )

    clone = req.copy()
    clone.extend_prompt([3])

    assert req.collapsed_input_ids == [1, 9]
    assert clone.collapsed_input_ids == [1, 9, 3]


def test_parse_handles_an_abort_with_no_output():
    """Test that an aborted request yields an empty result, not a crash.

    Aborts happen on every weight update, so this path is load bearing.
    """
    response = {
        "choices": [{"finish_reason": "abort", "token_ids": [], "logprobs": None}]
    }

    result = VLLMBackend().parse_generation_response(response)

    assert result.output_tokens == []
    assert result.output_logprobs == []
    assert result.stop_reason == "abort"


def test_media_placeholders_are_prohibited_at_the_sampler():
    """Test that reserved media tokens are banned, not merely filtered later.

    Filtering the response is too late for an interrupted generation: the abort
    loop appends each partial segment to both prompts before the client-side
    filter runs, so a sampled placeholder would reach the next request and fail
    media matching or the exact-token check.
    """
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    payload = (
        VLLMBackend()
        .build_generation_request(_vision_request(gconfig), with_lora=False, version=0)
        .payload
    )

    assert payload["sampling_params"]["bad_words"] == ["<|image_pad|>", "<|video_pad|>"]


def test_sampler_ban_follows_the_placeholder_the_processor_declares():
    """Test that the ban targets the model's own placeholder, not a fixed name.

    vLLM expands whatever ``hf_processor.image_token`` names, so a ban built
    from the same attribute cannot aim at the wrong token on a non-Qwen VLM.
    """

    class _Processor:
        image_token = "<pic>"
        video_token = "<vid>"

    class _Tokenizer:
        _ids = {"<pic>": 7, "<vid>": 8}

        def convert_tokens_to_ids(self, token):
            return self._ids.get(token)

        def convert_ids_to_tokens(self, token_id):
            for token, tid in self._ids.items():
                if tid == token_id:
                    return token
            return None

    req = _vision_request(GenerationHyperparameters(max_new_tokens=8))
    req.processor = _Processor()
    req.tokenizer = _Tokenizer()
    # This model's placeholder is <pic> (id 7), so its prompts carry that id --
    # the count guard reads the same declaration as the ban.
    req.input_ids = [11, 7, 7, 7, 7, 12]
    req.collapsed_input_ids = [11, 7, 12]

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["sampling_params"]["bad_words"] == ["<pic>", "<vid>"]


def test_media_request_rejected_when_no_placeholder_resolves():
    """Test that an unresolvable placeholder is refused, not silently accepted.

    Such a prompt cannot be checked against its media, and vLLM could not expand
    it either, so failing here beats an opaque render error on the server.
    """

    class _ForeignTokenizer:
        def convert_tokens_to_ids(self, token):
            return None

        def convert_ids_to_tokens(self, token_id):
            return None

    req = _vision_request(GenerationHyperparameters(max_new_tokens=8))
    req.tokenizer = _ForeignTokenizer()

    with pytest.raises(ValueError, match="collapsed media marker"):
        VLLMBackend().build_generation_request(req, with_lora=False, version=0)


def test_media_request_rejected_without_a_tokenizer():
    """Test that the placeholder guard refuses to run blind.

    Skipping validation when no tokenizer is reachable would let a prompt whose
    placeholder count disagrees with its media reach the server unchecked.
    """
    req = _vision_request(GenerationHyperparameters(max_new_tokens=8))
    req.tokenizer = None

    with pytest.raises(ValueError, match="needs a tokenizer"):
        VLLMBackend().build_generation_request(req, with_lora=False, version=0)


def test_placeholder_guard_accepts_a_tokenizer_from_the_processor():
    """Test that a request carrying only a processor is still validated."""

    class _Processor:
        image_token = "<|image_pad|>"
        tokenizer = _QwenLikeTokenizer()

    req = _vision_request(GenerationHyperparameters(max_new_tokens=8))
    req.tokenizer = None
    req.processor = _Processor()

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["sampling_params"]["bad_words"] == ["<|image_pad|>"]


def test_text_requests_do_not_ban_media_placeholders():
    """Test that the text path is not burdened with a media-only restriction."""
    req = ModelRequest(
        input_ids=[11], gconfig=GenerationHyperparameters(max_new_tokens=4)
    )

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert "bad_words" not in payload["sampling_params"]


def test_parse_rejects_tokens_without_logprobs():
    """Test that a response carrying tokens but no logprobs is an error.

    Behaviour logprobs are required training data. Only comparing lengths when
    logprobs are non-empty would let such a response through with the two
    sequences silently unmatched.
    """
    response = {
        "choices": [{"finish_reason": "stop", "token_ids": [5, 6], "logprobs": None}]
    }

    with pytest.raises(ValueError, match="parallel"):
        VLLMBackend().parse_generation_response(response)


def test_parse_accepts_an_abort_with_neither_tokens_nor_logprobs():
    """Test that an aborted request with no output is still valid.

    Both sequences are empty, so the parallel-length rule holds.
    """
    response = {
        "choices": [{"finish_reason": "abort", "token_ids": [], "logprobs": None}]
    }

    result = VLLMBackend().parse_generation_response(response)

    assert result.output_tokens == []
    assert result.output_logprobs == []
    assert result.stop_reason == "abort"


def test_collapsed_prompt_with_wrong_placeholder_count_fails_client_side():
    """Test that a mismatched placeholder count is caught before sending.

    vLLM replaces one placeholder per media item; a different count makes its
    renderer assert, which surfaces as an opaque HTTP 500 with the exact-token
    diagnostic never reached, because rendering fails first.
    """
    from transformers import AutoTokenizer  # noqa: F401  (documents intent)

    class _Tok:
        def convert_tokens_to_ids(self, token):
            return {"<|image_pad|>": IMAGE_PAD, "<|video_pad|>": 151656}.get(token)

        def convert_ids_to_tokens(self, token_id):
            return {IMAGE_PAD: "<|image_pad|>", 151656: "<|video_pad|>"}.get(token_id)

    req = _vision_request(GenerationHyperparameters(max_new_tokens=4))
    req.tokenizer = _Tok()
    req.collapsed_input_ids = [11, 12]  # no placeholder at all

    with pytest.raises(ValueError, match="media placeholder"):
        VLLMBackend().build_generation_request(req, with_lora=False, version=0)


def test_matching_placeholder_count_passes_the_guard():
    """Test that a correctly built collapsed prompt is not obstructed."""

    class _Tok:
        def convert_tokens_to_ids(self, token):
            return {"<|image_pad|>": IMAGE_PAD, "<|video_pad|>": 151656}.get(token)

        def convert_ids_to_tokens(self, token_id):
            return {IMAGE_PAD: "<|image_pad|>", 151656: "<|video_pad|>"}.get(token_id)

    req = _vision_request(GenerationHyperparameters(max_new_tokens=4))
    req.tokenizer = _Tok()

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["token_ids"] == req.collapsed_input_ids
