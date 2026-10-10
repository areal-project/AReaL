# SPDX-License-Identifier: Apache-2.0
"""Qwen thinking defaults and optional per-request prefix-cache isolation."""

import copy
import os
import uuid

from examples.swe.qwen38_flash_next.template_defaults import wrap_create


def wrap_isolated_cache(original):
    def build(self, *args, **kwargs):
        request = copy.copy(original(self, *args, **kwargs))
        request.payload = dict(request.payload)
        request.payload["cache_salt"] = uuid.uuid4().hex
        return request

    return build


def main():
    from areal.engine.sglang_remote import SGLangBackend
    from areal.experimental.openai.client import AsyncCompletionsWithReward
    from areal.experimental.openai.proxy import proxy_rollout_server

    AsyncCompletionsWithReward.create = wrap_create(AsyncCompletionsWithReward.create)
    if os.environ.get("QWEN_ISOLATE_REQUEST_CACHE") == "1":
        SGLangBackend.build_generation_request = wrap_isolated_cache(
            SGLangBackend.build_generation_request
        )
    proxy_rollout_server.main()


if __name__ == "__main__":
    main()
