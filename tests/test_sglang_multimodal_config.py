# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from areal.engine import sglang_remote


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
