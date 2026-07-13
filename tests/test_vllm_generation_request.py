from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.vllm_remote import VLLMBackend


def test_vllm_forwards_frequency_penalty_and_stop():
    """The vLLM backend must forward frequency_penalty and stop like the SGLang
    backend does; both are GenerationHyperparameters and are accepted by vLLM's
    OpenAI-compatible /v1/completions endpoint."""
    gconfig = GenerationHyperparameters(
        max_new_tokens=8, frequency_penalty=0.5, stop=["STOP"]
    )
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["frequency_penalty"] == 0.5
    assert payload["stop"] == ["STOP"]
