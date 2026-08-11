# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn

from areal.engine.weight_finite import (
    WEIGHT_FINITE_CHECK_ENV,
    WEIGHT_FINITE_CHUNK_NUMEL_ENV,
    check_named_tensors_finite,
    iter_module_named_tensors,
)


def test_weight_finite_check_disabled_does_not_consume_tensors(monkeypatch):
    """The disabled diagnostic adds no model traversal work."""
    monkeypatch.delenv(WEIGHT_FINITE_CHECK_ENV, raising=False)
    consumed = False

    def named_tensors():
        nonlocal consumed
        consumed = True
        yield "weight", torch.ones(1)

    report = check_named_tensors_finite(
        named_tensors(), stage="disabled", logger=logging.getLogger("test")
    )

    assert report is None
    assert consumed is False


def test_weight_finite_check_reports_finite_module_weights(monkeypatch):
    """Parameters and floating buffers are counted while integers are ignored."""
    monkeypatch.setenv(WEIGHT_FINITE_CHECK_ENV, "1")
    module = nn.Linear(3, 2)
    module.register_buffer("scale", torch.ones(2))
    module.register_buffer("indices", torch.arange(2))

    report = check_named_tensors_finite(
        iter_module_named_tensors(module),
        stage="actor_post_optimizer",
        version=4,
        logger=logging.getLogger("test"),
    )

    assert report is not None
    assert report.stage == "actor_post_optimizer"
    assert report.version == 4
    assert report.tensor_count == 3
    assert report.numel == module.weight.numel() + module.bias.numel() + 2


def test_weight_finite_check_names_nan_and_inf_tensors(monkeypatch):
    """A failed scan identifies tensors and separates NaN from Inf counts."""
    monkeypatch.setenv(WEIGHT_FINITE_CHECK_ENV, "true")
    named_tensors = [
        ("good", torch.ones(4)),
        ("bad_nan", torch.tensor([0.0, float("nan"), float("nan")])),
        ("bad_inf", torch.tensor([float("inf"), float("-inf")])),
    ]

    with pytest.raises(FloatingPointError) as exc_info:
        check_named_tensors_finite(
            named_tensors,
            stage="awex_writer_converted",
            version=12,
            logger=logging.getLogger("test"),
        )

    message = str(exc_info.value)
    assert "stage=awex_writer_converted version=12 rank=0" in message
    assert "bad_nan:shape=(3,),dtype=torch.float32,nan=2,inf=0" in message
    assert "bad_inf:shape=(2,),dtype=torch.float32,nan=0,inf=2" in message


def test_weight_finite_check_chunks_noncontiguous_tensor(monkeypatch):
    """Chunked scanning detects invalid values without flattening tensor storage."""
    monkeypatch.setenv(WEIGHT_FINITE_CHECK_ENV, "on")
    monkeypatch.setenv(WEIGHT_FINITE_CHUNK_NUMEL_ENV, "3")
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1)
    tensor[4, 2] = float("nan")
    assert tensor.is_contiguous() is False

    with pytest.raises(FloatingPointError, match="derived.w_kc"):
        check_named_tensors_finite(
            [("derived.w_kc", tensor)],
            stage="awex_reader_derived",
            logger=logging.getLogger("test"),
        )


def test_weight_finite_check_fails_when_another_rank_is_bad(monkeypatch):
    """A collective failure makes locally healthy ranks stop as well."""
    monkeypatch.setenv(WEIGHT_FINITE_CHECK_ENV, "1")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_rank", lambda: 3)

    def all_reduce(tensor, op, group):
        del op, group
        tensor.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    with pytest.raises(FloatingPointError, match="another distributed rank"):
        check_named_tensors_finite(
            [("local_weight", torch.ones(2))],
            stage="actor_post_optimizer",
            logger=logging.getLogger("test"),
            process_group=object(),
        )


def test_iter_module_named_tensors_includes_selected_derived_attrs():
    """SGLang plain tensor attributes can be scanned with registered weights."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.proj.w_kc = torch.ones(2, 2)
            self.proj.unrelated_cache = torch.ones(1)

    tensors = dict(
        iter_module_named_tensors(
            Model(),
            include_parameters=False,
            include_buffers=False,
            extra_tensor_attrs=("w_kc", "w_vc"),
        )
    )

    assert list(tensors) == ["proj.w_kc"]
    torch.testing.assert_close(
        tensors["proj.w_kc"], torch.ones(2, 2), rtol=0.0, atol=0.0
    )
