# SPDX-License-Identifier: Apache-2.0
"""CPU/Gloo regressions for FSDP's tensor-parallel model plan."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _fsdp_tp_boundary_worker(rank, rendezvous):
    import copy

    from torch import nn
    from torch.distributed.device_mesh import init_device_mesh

    from areal.engine.fsdp_utils.parallel import apply_non_moe_tp

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(47)
        model = nn.Module()
        model.config = SimpleNamespace(
            num_attention_heads=2, num_key_value_heads=2, model_type="qwen3"
        )
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(24, 8, dtype=torch.float64)
        model.model.norm = nn.LayerNorm(8, dtype=torch.float64)
        model.model.layers = nn.ModuleList()
        model.lm_head = nn.Linear(8, 24, bias=False, dtype=torch.float64)
        ref = copy.deepcopy(model)
        mesh = init_device_mesh("cpu", (2,))
        apply_non_moe_tp(model, model.config, SimpleNamespace(tp_size=2), mesh)
        tokens = torch.tensor([[2, 5, 7, 13, 17, 19, 21, 23]])
        # HF selects positions between final norm and the projection. This must
        # use global token positions, not positions in a local sequence shard.
        hidden = model.model.norm(model.model.embed_tokens(tokens))
        assert type(hidden) is torch.Tensor
        output = model.lm_head(hidden[:, 1:7:2, :])
        expected = ref.lm_head(
            ref.model.norm(ref.model.embed_tokens(tokens))[:, 1:7:2, :]
        )
        torch.testing.assert_close(
            output, expected[..., rank * 12 : (rank + 1) * 12], atol=1e-12, rtol=1e-12
        )
        output.square().sum().backward()
        expected.square().sum().backward()
        for (name, p), (_, r) in zip(model.named_parameters(), ref.named_parameters()):
            g = p.grad.full_tensor()
            torch.testing.assert_close(g, r.grad, atol=1e-10, rtol=1e-10, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_fsdp_tp_model_wrapper_slice_preserves_outputs_and_gradients(tmp_path):
    mp.spawn(
        _fsdp_tp_boundary_worker,
        args=(str(tmp_path / "boundary"),),
        nprocs=2,
        join=True,
    )


def _qwen_tp_gradient_worker(rank, rendezvous, use_wrapper):
    import copy

    from torch.distributed.device_mesh import init_device_mesh
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from areal.engine.fsdp_utils.parallel import apply_non_moe_tp

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(137)
        config = Qwen3Config(
            vocab_size=24,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attention_dropout=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).double()
        ref = copy.deepcopy(model)
        apply_non_moe_tp(
            model, config, SimpleNamespace(tp_size=2), init_device_mesh("cpu", (2,))
        )
        tokens = torch.tensor([[2, 5, 7, 13, 17, 19, 21, 23]])
        # Exercise the production backbone and TP projection directly. The
        # causal-LM wrapper's DTensor slicing issue is separate from Q/K norm
        # gradient reduction and must not hide this regression.
        if use_wrapper:
            output = model(tokens, use_cache=False).logits
            expected = ref(tokens, use_cache=False).logits
        else:
            output = model.lm_head(
                model.model(tokens, use_cache=False).last_hidden_state
            )
            expected = ref.lm_head(ref.model(tokens, use_cache=False).last_hidden_state)
        torch.testing.assert_close(
            output, expected[..., rank * 12 : (rank + 1) * 12], atol=2e-7, rtol=2e-6
        )
        # Head-sharded Q/K normalization must aggregate the
        # shared norm-weight gradients correctly in the base TP backend.
        output.square().sum().backward()
        expected.square().sum().backward()
        for (name, p), (_, r) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(
                p.grad.full_tensor(), r.grad, atol=3e-6, rtol=3e-5, msg=name
            )
        torch.optim.Adam(model.parameters(), lr=1e-3).step()
        torch.optim.Adam(ref.parameters(), lr=1e-3).step()
        for (name, p), (_, r) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(
                p.full_tensor(), r, atol=3e-7, rtol=3e-6, msg=name
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("use_wrapper", [False, True])
def test_fsdp_qwen_tp_all_parameter_gradients_match_unsharded_model(
    tmp_path, use_wrapper
):
    mp.spawn(
        _qwen_tp_gradient_worker,
        args=(str(tmp_path / "qwen"), use_wrapper),
        nprocs=2,
        join=True,
    )
