# SPDX-License-Identifier: Apache-2.0

import pytest

from examples.vlm.pacman.policy import TokenSetRequest

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest


def test_request_plugin_uses_exact_support_without_top_k_filtering():
    plugin = TokenSetRequest()
    plugin.__dict__["processor"] = "serialized-public-processor"
    request = ModelRequest(
        rid="action",
        input_ids=[1],
        gconfig=GenerationHyperparameters(
            max_new_tokens=1, temperature=0.7, top_p=1.0, stop_token_ids=[99]
        ),
        metadata={"allowed_token_ids": [3, 7]},
    )
    payload = {"sampling_params": {"top_k": 100}}
    plugin.build_request(request, payload)
    assert payload["sampling_params"]["top_k"] == -1
    assert payload["sampling_params"]["custom_params"] == {"allowed_token_ids": [3, 7]}
    assert payload["custom_logit_processor"] == "serialized-public-processor"
    request.metadata["allowed_token_ids"] = [3, 3]
    with pytest.raises(ValueError, match="distinct"):
        plugin.build_request(request, payload)
