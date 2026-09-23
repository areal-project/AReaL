# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.hybrid_optimizer import (
    install_hybrid_optimizer_checkpoint_compat,
    sync_loaded_hybrid_optimizer_state,
)
from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


def make_optimizer(param):
    module = pytest.importorskip(
        "megatron.core.optimizer.cpu_offloading.hybrid_optimizer"
    )
    optimizer = module.HybridDeviceOptimizer(
        [param],
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        lr=0.01,
    )
    install_hybrid_optimizer_checkpoint_compat(optimizer)
    return optimizer


def update(optimizer, param, value):
    param.grad = torch.full_like(param, value)
    optimizer.step()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for HDO")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hybrid_restore_preserves_masters_moments_and_next_update(dtype):
    param = torch.nn.Parameter(torch.linspace(0.1, 1, 32, device="cuda", dtype=dtype))
    optimizer = make_optimizer(param)
    for value in (0.2, -0.7, 0.3):
        update(optimizer, param, value)
    saved = copy.deepcopy(optimizer.state_dict())
    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = make_optimizer(restored_param)
    restored.load_state_dict(saved)
    for key in ("master_param", "exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            restored.state[restored_param][key],
            optimizer.state[param][key],
            rtol=0,
            atol=0,
        )
    assert restored.param_to_inner_param[restored_param].device.type == "cpu"
    update(optimizer, param, -0.4)
    update(restored, restored_param, -0.4)
    torch.testing.assert_close(restored_param, param, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for HDO")
def test_dp_reshardable_restore_uses_checkpoint_adam_step_and_inner_master():
    module = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")
    param = torch.nn.Parameter(torch.linspace(0.1, 1, 32, device="cuda"))
    optimizer = make_optimizer(param)
    for value in (0.2, -0.7, 0.3):
        update(optimizer, param, value)
    tensors = {
        key: value.detach().clone() for key, value in optimizer.state[param].items()
    }
    tensors.update(param=param.detach().clone(), padding=False)
    # DCP LocalNonpersistentObject retains the template's dummy Adam step.
    tensors["step"] = torch.tensor(1.0)
    restored_param = torch.nn.Parameter(torch.zeros_like(param))
    restored = make_optimizer(restored_param)
    update(restored, restored_param, 0.0)
    restored.param_groups[0]["step"] = 3
    distributed = object.__new__(module.DistributedOptimizer)
    distributed.optimizer = restored
    distributed.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    distributed.model_param_group_index_map = {restored_param: (0, 0)}
    distributed.gbuf_ranges = [{torch.float32: [{"param_map": {restored_param: None}}]}]
    with torch.no_grad():
        distributed.load_parameter_state_from_dp_reshardable(
            {0: {torch.float32: [[tensors]]}}
        )
    sync_loaded_hybrid_optimizer_state(restored)
    inner = restored.param_to_inner_param[restored_param]
    sub = restored.cpu_optimizers[0]
    assert sub.state[inner]["step"].item() == 3
    for key in ("exp_avg", "exp_avg_sq"):
        assert sub.state[inner][key] is restored.state[restored_param][key]
        torch.testing.assert_close(sub.state[inner][key], tensors[key], rtol=0, atol=0)
    update(optimizer, param, -0.4)
    update(restored, restored_param, -0.4)
    torch.testing.assert_close(restored_param, param, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimizer_roundtrip_preserves_cpu_ownership_and_adam_updates(monkeypatch):
    monkeypatch.delenv("AWEX_OPT_OFFLOAD_VIA_HDO", raising=False)
    params = [torch.nn.Parameter(torch.ones(4, device=d)) for d in ("cpu", "cuda")]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opts = [torch.optim.AdamW([p], lr=0.01) for p in params]
    ref_opts = [torch.optim.AdamW([p], lr=0.01) for p in refs]
    wrapped = [
        SimpleNamespace(optimizer=o, shard_fp32_from_float16_groups=[[p]])
        for o, p in zip(opts, params)
    ]
    adapter = MegatronWeightResidency(
        SimpleNamespace(
            optimizer=SimpleNamespace(chained_optimizers=wrapped),
            device=torch.device("cuda"),
        )
    )
    for _ in range(3):
        for p, r, o, ro in zip(params, refs, opts, ref_opts):
            p.grad = torch.full_like(p, 0.5)
            r.grad = torch.full_like(r, 0.5)
            o.step()
            ro.step()
        cpu_moment = opts[0].state[params[0]]["exp_avg"]
        adapter._offload_optimizer_states()
        assert all(p.device.type == "cpu" for p in params)
        adapter._reload_optimizer_states()
        adapter._reload_optimizer_states()
        assert opts[0].state[params[0]]["exp_avg"] is cpu_moment
        for p, r, o, ro in zip(params, refs, opts, ref_opts):
            assert p.device == r.device
            torch.testing.assert_close(p, r)
            for key in ("exp_avg", "exp_avg_sq"):
                actual, expected = o.state[p][key], ro.state[r][key]
                assert actual.device == expected.device
                torch.testing.assert_close(actual, expected)
