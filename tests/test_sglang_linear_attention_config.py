# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from areal.api import cli_args
from areal.api.cli_args import SGLangConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"linear_attn_decode_backend": "triton"},
        {"linear_attn_prefill_backend": "triton"},
        {
            "linear_attn_decode_backend": "triton",
            "linear_attn_prefill_backend": "flashinfer",
        },
    ],
)
def test_linear_attention_backends_reach_sglang_command(monkeypatch, overrides):
    monkeypatch.setattr(
        cli_args.pkg_version, "is_version_greater_or_equal", lambda *args: True
    )
    config = OmegaConf.merge(OmegaConf.structured(SGLangConfig), overrides)
    command = SGLangConfig.build_cmd(config, tp_size=1, base_gpu_id=0)

    for key, flag in (
        ("linear_attn_decode_backend", "--linear-attn-decode-backend"),
        ("linear_attn_prefill_backend", "--linear-attn-prefill-backend"),
    ):
        if key in overrides:
            assert command[command.index(flag) + 1] == overrides[key]
        else:
            assert flag not in command
