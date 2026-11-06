# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unbuffered_offload_preserves_shared_storage_and_optimizer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    module = torch.nn.Module()
    backing = torch.randn(48, dtype=torch.bfloat16, device=device)
    module.a = torch.nn.Parameter(backing[3:19].view(4, 4))
    module.b = torch.nn.Parameter(backing[24:40].view(4, 4))
    module.tied = module.a
    module.flat = torch.nn.Parameter(torch.ones(4, device=device))
    flat_key = module.flat.untyped_storage()._cdata
    storage = backing.untyped_storage()
    key = storage._cdata
    residency = MegatronWeightResidency(SimpleNamespace(model=[module]))
    optimizer = torch.optim.SGD([module.a, module.b], lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        (module.a.float().square().sum() + module.b.float().square().sum()).backward()
        optimizer.step()
        expected = backing.clone()
        versions = (module.a._version, module.b._version)
        residency._offload_unbuffered_weights([module], {flat_key})
        assert storage.nbytes() == 0
        assert len(residency._unbuffered_weights) == 1
        assert module.flat.untyped_storage().nbytes() > 0
        residency._reload_unbuffered_weights()
        assert module.a.untyped_storage()._cdata == key
        assert module.b.untyped_storage()._cdata == key
        assert module.tied is module.a
        assert versions == (module.a._version, module.b._version)
        torch.testing.assert_close(backing, expected, rtol=0, atol=0)
        assert not residency._unbuffered_weights


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("master_change", ["inplace", "replace"])
def test_expert_direct_release_restores_master_and_preserves_storage(
    device, master_change, monkeypatch
):
    """Release without master copies; restore exact bytes and reject stale sources."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    module = torch.nn.Module()
    # Signed zero, BF16 subnormals, normal values and rounding boundaries.
    master = torch.tensor(
        [0.0, -0.0, 2.0**-133, -(2.0**-133), 1.00390625, 1.01171875, 2.0, -3.0]
    )
    module.expert = torch.nn.Parameter(master.to(device).bfloat16().view(2, 4))
    module.expert.allreduce = False
    module.other_expert = torch.nn.Parameter(
        torch.ones(2, 4, dtype=torch.bfloat16, device=device)
    )
    module.other_expert.allreduce = False
    module.tied = torch.nn.Parameter(
        torch.ones(2, 4, dtype=torch.bfloat16, device=device)
    )
    module.alias = module.tied
    module.tied.allreduce = False
    module.buffer_alias = torch.nn.Parameter(
        torch.ones(2, 4, dtype=torch.bfloat16, device=device)
    )
    module.buffer_alias.allreduce = False
    module.register_buffer("shared", module.buffer_alias.detach())
    owner = SimpleNamespace(
        state={
            p: {"master_param": p.detach().float().cpu()} for p in module.parameters()
        },
    )
    owner.state[module.expert]["master_param"] = master
    drains = []
    owner.drain = lambda: drains.append(True)
    owner.get_full_parameter_master = lambda p: owner.state.get(p, {}).get(
        "master_param"
    )
    optimizer = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=owner),
            SimpleNamespace(optimizer=SimpleNamespace()),
        ],
    )
    residency = MegatronWeightResidency(
        SimpleNamespace(model=[module], optimizer=optimizer)
    )

    def forbid_master_copy(*args, **kwargs):
        raise AssertionError("offload must not copy or convert FP32 masters")

    for cycle in range(2):
        # The master is authoritative even if a disposable BF16 copy differs.
        module.other_expert.data.fill_(2)
        saved = [
            (
                owner.state[p]["master_param"].to(device).bfloat16()
                if p is module.other_expert
                else p.detach()
            )
            .view(torch.int16)
            .cpu()
            .clone()
            for p in module.parameters()
        ]
        identities = [
            (p.untyped_storage()._cdata, p._version) for p in module.parameters()
        ]
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", forbid_master_copy)
            residency._offload_unbuffered_weights([module], set())
        assert len(drains) == cycle + 1
        assert len(residency._master_backed_weights) == 2
        assert len(residency._unbuffered_weights) == 2
        assert all(p.untyped_storage().nbytes() == 0 for p in module.parameters())
        residency._reload_unbuffered_weights()
        assert identities == [
            (p.untyped_storage()._cdata, p._version) for p in module.parameters()
        ]
        for expected, param in zip(saved, module.parameters(), strict=True):
            torch.testing.assert_close(
                param.detach().view(torch.int16).cpu(), expected, rtol=0, atol=0
            )
    residency._offload_unbuffered_weights([module], set())
    if master_change == "inplace":
        master.add_(1)
    else:
        owner.state[module.expert]["master_param"] = master.clone()
    with pytest.raises(RuntimeError, match="master changed"):
        residency._reload_unbuffered_weights()
    assert all(p.untyped_storage().nbytes() == 0 for p in module.parameters())
