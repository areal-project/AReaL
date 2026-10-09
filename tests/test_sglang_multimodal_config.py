# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from areal.api import ModelRequest
from areal.api.cli_args import GenerationHyperparameters
from areal.engine import sglang_remote


@pytest.mark.parametrize(
    ("input_ids", "image_count", "image_token_id", "expected"),
    [
        ([10, 99, 99, 99, 11, 20, 21], 1, 99, [10, 99, 11, 20, 21]),
        (
            [10, 99, 99, 11, 20, 21, 10, 99, 99, 99, 11],
            2,
            99,
            [10, 99, 11, 20, 21, 10, 99, 11],
        ),
        ([10, 99, 11, 20, 21], 1, 99, [10, 99, 11, 20, 21]),
        ([10, 0, 0, 11], 1, 0, [10, 0, 11]),
    ],
)
def test_sglang_image_request_preserves_expanded_training_tokens_and_budget(
    input_ids, image_count, image_token_id, expected
):
    original = input_ids.copy()
    images = [f"image-{index}" for index in range(image_count)]
    budget = 256 - len(original)
    req = ModelRequest(
        input_ids=input_ids,
        image_data=images,
        processor=SimpleNamespace(image_token_id=image_token_id),
        gconfig=GenerationHyperparameters(max_new_tokens=budget),
    )

    request = sglang_remote.SGLangBackend().build_generation_request(
        req, with_lora=False, version=0
    )

    assert request.payload["input_ids"] == expected
    assert request.payload["input_ids"] is not req.input_ids
    assert req.input_ids == original
    assert request.payload["image_data"] == images
    assert request.payload["sampling_params"]["max_new_tokens"] == budget


@pytest.mark.parametrize(
    ("images", "processor", "input_ids"),
    [
        (None, SimpleNamespace(image_token_id=99), [10, 99, 99, 11]),
        ([], SimpleNamespace(image_token_id=99), [10, 99, 99, 11]),
        (["image"], None, [10, 99, 99, 11]),
        (["image"], SimpleNamespace(), [10, 99, 99, 11]),
        (["image"], SimpleNamespace(image_token_id=None), [10, 99, 99, 11]),
        (["image"], SimpleNamespace(image_token_id="99"), [10, 99, 99, 11]),
        (["image"], SimpleNamespace(image_token_id=99), [10, 11]),
        (["image", "image"], SimpleNamespace(image_token_id=99), [10, 99, 99, 11]),
        (["image"], SimpleNamespace(image_token_id=99), [10, 99, 99, 11, 99]),
    ],
)
def test_sglang_image_request_keeps_tokens_when_placeholder_contract_is_unknown(
    images, processor, input_ids
):
    req = ModelRequest(input_ids=input_ids, image_data=images, processor=processor)

    request = sglang_remote.SGLangBackend().build_generation_request(
        req, with_lora=False, version=0
    )

    assert request.payload["input_ids"] == input_ids
    assert request.payload["input_ids"] is not input_ids


@pytest.mark.parametrize(
    ("skip_tokenizer_init", "warning_count"),
    [(False, 1), (True, 0)],
)
def test_sglang_multimodal_launch_warns_without_skip_tokenizer_init(
    monkeypatch, skip_tokenizer_init, warning_count
):
    """Multimodal launch should warn, but remain allowed, with tokenizer enabled."""
    warning = MagicMock()
    monkeypatch.setattr(sglang_remote.logger, "warning", warning)
    monkeypatch.setattr(
        sglang_remote.SGLangConfig,
        "build_cmd_from_args",
        lambda _args: ["sglang-server"],
    )
    popen = MagicMock()
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", popen)

    backend = sglang_remote.SGLangBackend()
    backend.launch_server(
        {
            "enable_multimodal": True,
            "skip_tokenizer_init": skip_tokenizer_init,
        }
    )

    assert warning.call_count == warning_count
    popen.assert_called_once()
