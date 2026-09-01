# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import types

from areal.engine.megatron_utils.mindspeed_args_patch import (
    ensure_mindspeed_args_sanitized,
    sanitize_get_full_args,
)


def test_sanitize_get_full_args_removes_invalid_namespace_keys():
    args = argparse.Namespace(valid_name=1)
    vars(args).update({"": True, "not-valid": 2, 3: "invalid"})

    sanitized = sanitize_get_full_args(lambda: args)()

    assert vars(sanitized) == {"valid_name": 1}


def test_ensure_mindspeed_args_sanitized_patches_shared_accessor(monkeypatch):
    args = argparse.Namespace(valid_name=1)
    vars(args)[""] = True
    args_utils = types.ModuleType("megatron_adaptor.utils.args_utils")

    def original_get_full_args():
        return args

    args_utils.get_full_args = original_get_full_args
    utils = types.ModuleType("megatron_adaptor.utils")
    utils.args_utils = args_utils
    adaptor = types.ModuleType("megatron_adaptor")
    adaptor.utils = utils
    transformer_config = types.ModuleType(
        "megatron_adaptor.patches.megatron.transformer_config"
    )
    transformer_config.get_full_args = original_get_full_args
    mindspeed_args = types.ModuleType("mindspeed.args_utils")
    mindspeed_args.get_full_args = original_get_full_args
    mindspeed_adaptor = types.ModuleType("mindspeed.megatron_adaptor")
    mindspeed_adaptor.get_full_args = original_get_full_args
    monkeypatch.setitem(sys.modules, "megatron_adaptor", adaptor)
    monkeypatch.setitem(sys.modules, "megatron_adaptor.utils", utils)
    monkeypatch.setitem(sys.modules, "megatron_adaptor.utils.args_utils", args_utils)
    monkeypatch.setitem(
        sys.modules,
        "megatron_adaptor.patches.megatron.transformer_config",
        transformer_config,
    )
    monkeypatch.setitem(sys.modules, "mindspeed.args_utils", mindspeed_args)
    monkeypatch.setitem(sys.modules, "mindspeed.megatron_adaptor", mindspeed_adaptor)

    assert ensure_mindspeed_args_sanitized()
    assert vars(args_utils.get_full_args()) == {"valid_name": 1}
    assert transformer_config.get_full_args is args_utils.get_full_args
    assert mindspeed_args.get_full_args is args_utils.get_full_args
    assert mindspeed_adaptor.get_full_args is args_utils.get_full_args
    assert ensure_mindspeed_args_sanitized()
