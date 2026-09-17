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


def test_vllm_forwards_explicit_seed():
    """The V1 vLLM backend forwards an explicitly configured sampling seed."""
    req = ModelRequest(
        input_ids=[11, 12],
        gconfig=GenerationHyperparameters(max_new_tokens=8, seed=12345),
    )

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert payload["seed"] == 12345


def test_vllm_omits_seed_when_unset():
    """The V1 vLLM backend leaves seed selection to vLLM when it is unset."""
    req = ModelRequest(
        input_ids=[11, 12],
        gconfig=GenerationHyperparameters(max_new_tokens=8),
    )

    payload = (
        VLLMBackend().build_generation_request(req, with_lora=False, version=0).payload
    )

    assert "seed" not in payload
