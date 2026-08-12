# SPDX-License-Identifier: Apache-2.0
"""Tests for the fp32 lm-head forward patch."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from areal.models.mcore.registry import (
    _enable_fp32_lm_head_forward,
    _fp32_lm_head_forward,
    _is_lm_head_module_name,
)


class TestLmHeadModuleName:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("output_layer", True),
            ("lm_head", True),
            ("decoder.output_layer", True),
            ("module.module.lm_head", True),
            ("output_layer_extra", False),
            ("decoder.layers.0.mlp", False),
            ("", False),
        ],
    )
    def test_matches_only_lm_head_modules(self, name, expected):
        assert _is_lm_head_module_name(name) is expected


class _Head(torch.nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 3))
        self.bias = None
        self.skip_bias_add = False
        self.sequence_parallel = False
        self.explicit_expert_comm = False
        self.gather_output = False
        self.tp_group = tp_group


class TestForwardForwardsTheTensorParallelGroup:
    def test_collectives_receive_the_modules_tp_group(self):
        group = object()
        head = _Head(group)
        seen = {}

        def copy_to(input_, group=None):
            seen["copy_to"] = group
            return input_

        def gather_from(input_, group=None):
            seen["gather_from"] = group
            return input_

        head.gather_output = True
        with mock.patch(
            "areal.models.mcore.registry.tensor_parallel",
            SimpleNamespace(
                copy_to_tensor_model_parallel_region=copy_to,
                gather_from_tensor_model_parallel_region=gather_from,
                gather_from_sequence_parallel_region=lambda *a, **k: a[0],
            ),
        ):
            _fp32_lm_head_forward(head, torch.ones(2, 3))

        assert seen["copy_to"] is group, (
            "copy_to_tensor_model_parallel_region ran on the default group; "
            "mcore forwards self.tp_group here"
        )
        assert seen["gather_from"] is group, (
            "gather_from_tensor_model_parallel_region ran on the default group; "
            "mcore forwards self.tp_group here"
        )

    def test_output_is_float32(self):
        head = _Head(None)
        with mock.patch(
            "areal.models.mcore.registry.tensor_parallel",
            SimpleNamespace(
                copy_to_tensor_model_parallel_region=lambda i, group=None: i,
                gather_from_tensor_model_parallel_region=lambda i, group=None: i,
                gather_from_sequence_parallel_region=lambda *a, **k: a[0],
            ),
        ):
            output, _bias = _fp32_lm_head_forward(head, torch.ones(2, 3).bfloat16())

        assert output.dtype is torch.float32


class TestEnablePatch:
    def test_disabled_patches_nothing(self):
        head = _Head(None)
        assert _enable_fp32_lm_head_forward([head], enabled=False) == 0

    def test_patch_is_idempotent(self):
        model = torch.nn.Module()
        model.add_module("output_layer", _Head(None))

        first = _enable_fp32_lm_head_forward([model], enabled=True)
        second = _enable_fp32_lm_head_forward([model], enabled=True)

        assert first == 1
        assert second == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
