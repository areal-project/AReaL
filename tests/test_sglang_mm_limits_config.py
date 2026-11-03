# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from omegaconf import OmegaConf

from areal.api import cli_args
from areal.api.cli_args import SGLangConfig


@pytest.fixture(autouse=True)
def installed_sglang(monkeypatch):
    monkeypatch.setattr(
        cli_args.pkg_version, "is_version_greater_or_equal", lambda *args: True
    )


@pytest.mark.parametrize("as_dataclass", [False, True])
@pytest.mark.parametrize(
    "limits",
    [
        None,
        {},
        {"image": 64, "video": 1, "audio": 0},
        {"image": 64, "video": 1, "audio": 5},
    ],
)
def test_mm_limits_yaml_reaches_cli_as_json(limits, as_dataclass):
    config = OmegaConf.merge(
        OmegaConf.structured(SGLangConfig),
        OmegaConf.create(
            "sglang:\n  limit_mm_data_per_request: " + json.dumps(limits)
        ).sglang,
    )
    if as_dataclass:
        config = OmegaConf.to_object(config)
    argv = SGLangConfig.build_cmd(config, tp_size=1, base_gpu_id=0)
    flag = "--limit-mm-data-per-request"
    if limits is None:
        assert flag not in argv
    else:
        assert json.loads(argv[argv.index(flag) + 1]) == limits


@pytest.mark.parametrize(
    "limits",
    [{"image": -1}, {"image": True}, {"image": 1.5}, {"images": 4}, "image=64"],
)
def test_mm_limits_invalid_values_fail_before_server_launch(limits):
    with pytest.raises(ValueError, match="limit_mm_data_per_request"):
        SGLangConfig(limit_mm_data_per_request=limits)


def test_mm_limits_dictconfig_validation_rejects_negative_count():
    config = OmegaConf.structured(SGLangConfig)
    config.limit_mm_data_per_request = {"image": -1}
    with pytest.raises(ValueError, match="limit_mm_data_per_request"):
        SGLangConfig.build_args(config, tp_size=1, base_gpu_id=0)
