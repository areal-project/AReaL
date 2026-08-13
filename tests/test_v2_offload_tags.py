"""Tests for backend-level offload tag plumbing."""

import pytest

from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend


class TestSGLangOffloadTags:
    def test_offload_without_tags_releases_everything(self):
        req = SGLangBridgeBackend().get_offload_request()
        assert req.endpoint == "/release_memory_occupation"
        assert req.payload == {}

    def test_offload_forwards_requested_tags(self):
        req = SGLangBridgeBackend().get_offload_request(tags=["kv_cache"])
        assert req.endpoint == "/release_memory_occupation"
        assert req.payload == {"tags": ["kv_cache"]}

    def test_offload_is_symmetric_with_onload(self):
        backend = SGLangBridgeBackend()
        tags = ["kv_cache", "cuda_graph"]
        assert (
            backend.get_offload_request(tags=tags).payload
            == backend.get_onload_request(tags=tags).payload
        )


class TestVLLMOffloadTags:
    def test_offload_without_tags_sleeps(self):
        req = VLLMBridgeBackend().get_offload_request()
        assert req.endpoint == "/sleep"

    def test_partial_offload_is_rejected_rather_than_silently_ignored(self):
        # vLLM's /sleep takes no tag selector, so a partial request must fail
        # loudly instead of releasing everything behind the caller's back.
        with pytest.raises(NotImplementedError, match="tags"):
            VLLMBridgeBackend().get_offload_request(tags=["kv_cache"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
