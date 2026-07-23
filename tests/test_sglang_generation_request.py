from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.sglang_remote import SGLangBackend


def test_sglang_forwards_sampling_seed_when_set():
    """When sampling_seed is set, it must reach sample_params so SGLang's seeded
    Gumbel sampler (multinomial_with_seed) can consume it."""
    gconfig = GenerationHyperparameters(max_new_tokens=8, sampling_seed=12345)
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        SGLangBackend()
        .build_generation_request(req, with_lora=False, version=0)
        .payload
    )

    assert payload["sampling_params"]["sampling_seed"] == 12345


def test_sglang_omits_sampling_seed_by_default():
    """Default (None) must send no seed field at all, so existing deployments and
    requests are byte-for-byte unaffected."""
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        SGLangBackend()
        .build_generation_request(req, with_lora=False, version=0)
        .payload
    )

    assert "sampling_seed" not in payload["sampling_params"]
