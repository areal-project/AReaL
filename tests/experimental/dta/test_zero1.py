# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from areal.api.cli_args import OptimizerConfig
from areal.experimental.dta import wrapper as dta_wrapper


class TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_args = SimpleNamespace(enable_weight_tying=True)
        self.tok_embeddings = nn.Embedding(4, 3)
        self.output = nn.Linear(3, 4, bias=False)


class TinyDTAWrapper:
    def __init__(self, model: nn.Module) -> None:
        self.engine = SimpleNamespace(model=model, model_parts=[])


def test_apply_zero1_ties_embedding_and_output_weight() -> None:
    model = TinyTiedModel()

    dta_wrapper.DTAWrapper.apply_zero1(TinyDTAWrapper(model))

    assert model.output.weight is model.tok_embeddings.weight
    param_names = dict(model.named_parameters())
    assert "tok_embeddings.weight" in param_names
    assert "output.weight" not in param_names
    loss = model.output(model.tok_embeddings(torch.tensor([0, 1]))).sum()
    loss.backward()
    assert model.output.weight.grad is model.tok_embeddings.weight.grad


def test_create_zero1_optimizer_receives_tied_parameter_once(monkeypatch) -> None:
    model = TinyTiedModel()
    dta_wrapper.DTAWrapper.apply_zero1(TinyDTAWrapper(model))
    captured: dict[str, object] = {}

    class FakeZeroRedundancyOptimizer:
        def __init__(self, params, **kwargs) -> None:
            captured["params"] = list(params)
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        dta_wrapper, "ZeroRedundancyOptimizer", FakeZeroRedundancyOptimizer
    )

    wrapper = TinyDTAWrapper(model)
    wrapper.engine.optimizer_config = OptimizerConfig(type="adam", lr=1e-3)
    wrapper.engine.data_parallel_group = object()
    wrapper.engine._get_all_parameters = lambda: list(model.parameters())

    optimizer = dta_wrapper.DTAWrapper.create_optimizer(
        wrapper,
    )

    assert isinstance(optimizer, FakeZeroRedundancyOptimizer)
    assert captured["params"].count(model.tok_embeddings.weight) == 1
    assert captured["params"].count(model.output.weight) == 1
