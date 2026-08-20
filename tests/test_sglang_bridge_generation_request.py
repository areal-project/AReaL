from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


def test_sglang_bridge_forwards_sampling_seed_when_set():
    """SGLangBridgeBackend mirrors SGLangBackend's sampling_seed forwarding (its own
    docstring says it "Mirrors the relevant subset of ... SGLangBackend") -- the v2
    data-proxy path must not silently diverge from the v1 remote-engine path."""
    gconfig = GenerationHyperparameters(max_new_tokens=8, sampling_seed=12345)
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        SGLangBridgeBackend()
        .build_generation_request(req, with_lora=False, version=0)
        .payload
    )

    assert payload["sampling_params"]["sampling_seed"] == 12345


def test_sglang_bridge_omits_sampling_seed_by_default():
    """Default (None) must send no seed field at all, so existing deployments and
    requests are byte-for-byte unaffected."""
    gconfig = GenerationHyperparameters(max_new_tokens=8)
    req = ModelRequest(input_ids=[11, 12], gconfig=gconfig)

    payload = (
        SGLangBridgeBackend()
        .build_generation_request(req, with_lora=False, version=0)
        .payload
    )

    assert "sampling_seed" not in payload["sampling_params"]
