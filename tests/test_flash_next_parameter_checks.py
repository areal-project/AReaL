# SPDX-License-Identifier: Apache-2.0
"""Negative acceptance checks for ownership and optimizer identity isolation."""

from types import SimpleNamespace

import pytest
import torch

from examples.mtp.qwen38_flash_next.parameter_checks import (
    verify_mtp_optimizer_membership,
)


def _model():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2, bias=False)
    model.mtp = torch.nn.Linear(2, 2, bias=False)
    model.backbone.requires_grad_(False)
    return model


def test_non_mtp_accidentally_unfrozen_is_rejected():
    model = _model()
    model.backbone.requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(AssertionError, match="trainability disagrees"):
        verify_mtp_optimizer_membership([model], optimizer)


def test_extra_frozen_parameter_in_optimizer_is_rejected():
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(AssertionError, match="extra=1"):
        verify_mtp_optimizer_membership([model], optimizer)


def test_shared_mtp_backbone_alias_remains_frozen():
    model = _model()
    model.mtp.shared = model.backbone.weight
    optimizer = torch.optim.SGD([model.mtp.weight], lr=0.1)
    counts = verify_mtp_optimizer_membership([model], optimizer)
    assert counts == {
        "frozen_parameters": 1,
        "mtp_parameters": 1,
        "optimizer_model_parameters": 1,
    }


def test_mcore_master_shards_are_compared_through_original_references():
    model = _model()
    optimizer = SimpleNamespace(
        model_float16_groups=[[model.mtp.weight]], model_fp32_groups=[]
    )
    assert verify_mtp_optimizer_membership([model], optimizer)["mtp_parameters"] == 1


def test_pipeline_stage_without_mtp_accepts_empty_optimizer():
    model = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    optimizer = SimpleNamespace(param_groups=[])
    assert verify_mtp_optimizer_membership([model], optimizer)["mtp_parameters"] == 0
