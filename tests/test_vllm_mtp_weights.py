# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.vllm_ext.npu_weight_reload import NpuReloadModel, weight_update_models

pytestmark = [pytest.mark.npu, pytest.mark.vllm]


@pytest.mark.parametrize("moe", [False, True])
def test_native_qwen_draft_loader_refreshes_mtp_and_shared_weights(moe):
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP

    model_class = Qwen3_5MoeMTP if moe else Qwen3_5MTP
    draft = model_class.__new__(model_class)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.fc = torch.nn.Linear(4, 2, bias=False)
    draft.model.embed_tokens = torch.nn.Embedding(8, 2)
    draft.lm_head = torch.nn.Linear(2, 8, bias=False)
    target = torch.nn.Module()
    target.load_weights = lambda weights: set()
    runner = SimpleNamespace(
        model=target,
        model_config=object(),
        drafter=SimpleNamespace(model=draft),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp", draft_model_config=object()
            )
        ),
    )
    reload_model = NpuReloadModel(*(m for m, _ in weight_update_models(runner)))
    pointers = {n: p.data_ptr() for n, p in draft.named_parameters()}
    for value in (3.0, 7.0):
        reload_model.load_weights(
            [
                ("mtp.fc.weight", torch.full((2, 4), value)),
                ("model.language_model.embed_tokens.weight", torch.full((8, 2), value)),
                ("lm_head.weight", torch.full((8, 2), value)),
                ("model.language_model.layers.0.other.weight", torch.ones(1)),
            ],
        )
        for name, param in draft.named_parameters():
            assert param.data_ptr() == pointers[name]
            torch.testing.assert_close(
                param, torch.full_like(param, value), rtol=0, atol=0
            )
