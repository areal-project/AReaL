# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import distrib_optimizer as distrib_optimizer_module
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from areal.engine.megatron_utils.gpu_staged_optimizer import (
    AdamWCPUSlabs,
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
    SlotStateMachine,
    bind_gpu_staged_adamw,
    get_megatron_optimizer_with_gpu_staged_adamw,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    abort_managed_checkpoint_load,
    begin_managed_async_checkpoint_save,
    begin_managed_checkpoint_load,
    bind_managed_async_checkpoint_request,
    complete_managed_async_checkpoint_save,
    complete_managed_checkpoint_load,
    prepare_managed_checkpoint_save,
    reset_managed_optimizer_from_model,
)

CUDA_AVAILABLE = torch.cuda.is_available()


def _tiny_config(buffer_count: int = 2, bucket_numel: int = 16):
    return GPUStagedAdamWConfig(
        buffer_count=buffer_count,
        bucket_size_mb=bucket_numel * 4 / (1024 * 1024),
    )


def test_cpu_slabs_are_contiguous_pinned_and_zero_initialized() -> None:
    """Each state kind occupies one flat pinned allocation."""
    slabs = AdamWCPUSlabs.allocate(37)

    assert slabs.master.shape == (37,)
    assert slabs.master.is_pinned()
    assert slabs.exp_avg.is_pinned()
    assert slabs.exp_avg_sq.is_pinned()
    assert slabs.master.dtype is torch.float32
    torch.testing.assert_close(slabs.exp_avg, torch.zeros(37), rtol=0.0, atol=0.0)
    torch.testing.assert_close(slabs.exp_avg_sq, torch.zeros(37), rtol=0.0, atol=0.0)


def test_slot_state_machine_waits_before_reuse_and_drain() -> None:
    """Pending D2H is waited exactly on reuse or drain."""
    waited: list[int] = []
    machine = SlotStateMachine(2, waited.append)

    machine.acquire(0)
    machine.mark_d2h_pending(0)
    machine.mark_d2h_pending(1)
    machine.acquire(0)

    assert waited == [0]
    assert machine.phases == ("FREE", "D2H_PENDING")
    machine.drain()
    assert waited == [0, 1]
    assert machine.phases == ("FREE", "FREE")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_bind_builds_cpu_state_views_and_bounded_units() -> None:
    """Binding initializes slab-backed state without persistent CUDA state."""
    params = [
        torch.nn.Parameter(torch.randn(11, device="cuda", dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(13, device="cuda", dtype=torch.bfloat16)),
    ]
    optimizer = GPUStagedAdamW(params, staged_config=_tiny_config(bucket_numel=8))

    optimizer.bind_owned_params(optimizer.param_groups)

    assert optimizer.residency == "CPU_RESIDENT"
    assert [unit.numel for unit in optimizer.units] == [8, 8, 8]
    assert optimizer.cpu_slabs is not None
    expected_offsets = [0, params[0].numel()]
    for param, expected_offset in zip(params, expected_offsets):
        state = optimizer.state[param]
        assert set(state) == {"master_param", "exp_avg", "exp_avg_sq"}
        assert all(tensor.device.type == "cpu" for tensor in state.values())
        assert all(tensor.is_pinned() for tensor in state.values())
        assert state["master_param"].untyped_storage().data_ptr() == (
            optimizer.cpu_slabs.master.untyped_storage().data_ptr()
        )
        assert state["master_param"].storage_offset() == expected_offset
        torch.testing.assert_close(
            state["master_param"], param.detach().cpu().float(), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            state["exp_avg"], torch.zeros_like(state["exp_avg"]), rtol=0.0, atol=0.0
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_multiple_steps_match_fp32_master_adamw() -> None:
    """DP=1 shard updates match ordinary FP32-master AdamW over many steps."""
    torch.manual_seed(1234)
    initial = torch.randn(41, device="cuda", dtype=torch.bfloat16)
    staged_param = torch.nn.Parameter(initial.clone())
    baseline_param = torch.nn.Parameter(initial.float())
    kwargs = {
        "lr": 3e-3,
        "betas": (0.8, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.07,
    }
    staged = GPUStagedAdamW(
        [staged_param], staged_config=_tiny_config(bucket_numel=9), **kwargs
    )
    staged.bind_owned_params(staged.param_groups)
    baseline = torch.optim.AdamW([baseline_param], **kwargs)

    for _ in range(7):
        grad = torch.randn_like(staged_param)
        staged_param.decoupled_grad = grad
        baseline_param.grad = grad.float()
        staged.step()
        baseline.step()
        staged.drain()

        state = staged.state[staged_param]
        baseline_state = baseline.state[baseline_param]
        torch.testing.assert_close(
            state["master_param"], baseline_param.detach().cpu(), rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            state["exp_avg"], baseline_state["exp_avg"].cpu(), rtol=2e-6, atol=2e-6
        )
        torch.testing.assert_close(
            state["exp_avg_sq"],
            baseline_state["exp_avg_sq"].cpu(),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            staged_param, baseline_param.detach().bfloat16(), rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("use_non_default_stream", [False, True])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_step_waits_for_caller_stream_gradient_write(
    use_non_default_stream: bool,
) -> None:
    """Delayed caller-stream gradients are visible to all staging streams."""
    torch.manual_seed(111)
    initial = torch.randn(37, device="cuda", dtype=torch.bfloat16)
    staged_param = torch.nn.Parameter(initial.clone())
    baseline_param = torch.nn.Parameter(initial.float())
    kwargs = {
        "lr": 2e-3,
        "betas": (0.7, 0.91),
        "eps": 1e-6,
        "weight_decay": 0.03,
    }
    staged = GPUStagedAdamW(
        [staged_param], staged_config=_tiny_config(bucket_numel=8), **kwargs
    )
    staged.bind_owned_params(staged.param_groups)
    baseline = torch.optim.AdamW([baseline_param], **kwargs)
    expected_grad = torch.linspace(
        -1, 1, staged_param.numel(), device="cuda", dtype=torch.bfloat16
    )
    delayed_grad = torch.zeros_like(expected_grad)
    caller_stream = (
        torch.cuda.Stream() if use_non_default_stream else torch.cuda.current_stream()
    )

    with torch.cuda.stream(caller_stream):
        torch.cuda._sleep(10_000_000)
        delayed_grad.copy_(expected_grad)
        staged_param.decoupled_grad = delayed_grad
        staged.step()

    caller_stream.synchronize()
    staged.drain()
    baseline_param.grad = expected_grad.float()
    baseline.step()
    staged_state = staged.state[staged_param]
    baseline_state = baseline.state[baseline_param]
    torch.testing.assert_close(
        staged_param, baseline_param.detach().bfloat16(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        staged_state["master_param"],
        baseline_param.detach().cpu(),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        staged_state["exp_avg"],
        baseline_state["exp_avg"].cpu(),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        staged_state["exp_avg_sq"],
        baseline_state["exp_avg_sq"].cpu(),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_mcore_distributed_optimizer_dp1_preserves_step_and_sync_order(
    monkeypatch,
) -> None:
    """MCore's real DP-shard and step path matches AdamW and syncs last."""

    class _RankOneGroup:
        @staticmethod
        def rank() -> int:
            return 0

        @staticmethod
        def size() -> int:
            return 1

    class _ModelChunk:
        def __init__(self, ddp_config):
            self.ddp_config = ddp_config
            self.expected_param = None
            self.param_sync_count = 0

        def start_param_sync(self) -> None:
            assert self.expected_param is not None
            torch.testing.assert_close(
                model_param, self.expected_param, rtol=0.0, atol=0.0
            )
            self.param_sync_count += 1

    monkeypatch.setattr(distrib_optimizer_module, "partition_buckets", lambda _: [])
    torch.manual_seed(4321)
    model_param = torch.nn.Parameter(
        torch.randn(41, device="cuda", dtype=torch.bfloat16)
    )
    baseline_param = torch.nn.Parameter(model_param.detach().float().clone())
    adam_kwargs = {
        "lr": 3e-3,
        "betas": (0.8, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.07,
    }
    inner = GPUStagedAdamW(
        [
            {
                "params": [model_param],
                "lr_mult": 1.0,
                "wd_mult": 1.0,
                "is_decoupled_lr": False,
            }
        ],
        staged_config=_tiny_config(bucket_numel=9),
        **adam_kwargs,
    )
    group = _RankOneGroup()
    ddp_config = SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=False)
    bucket = SimpleNamespace(
        grad_data=torch.empty_like(model_param),
        param_data=model_param.detach(),
        offset=0,
        numel_unpadded=model_param.numel(),
    )
    buffer = SimpleNamespace(
        param_dtype=torch.bfloat16,
        grad_dtype=torch.bfloat16,
        buckets=[bucket],
        param_index_map={model_param: (0, model_param.numel(), 0)},
        data_parallel_group=group,
        params=[model_param],
    )
    model_chunk = _ModelChunk(ddp_config)
    config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=adam_kwargs["lr"],
        min_lr=0.0,
        weight_decay=adam_kwargs["weight_decay"],
        adam_beta1=adam_kwargs["betas"][0],
        adam_beta2=adam_kwargs["betas"][1],
        adam_eps=adam_kwargs["eps"],
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_grads_dtype=torch.bfloat16,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=0.0,
    )
    optimizer = DistributedOptimizer(
        inner,
        config,
        grad_scaler=None,
        init_state_fn=None,
        model_chunks=[model_chunk],
        per_model_buffers={0: [buffer]},
        data_parallel_group=group,
        data_parallel_group_gloo=None,
        data_parallel_group_idx=0,
        distributed_optimizer_instance_id=0,
    )
    assert all(
        main_param is None
        for param_group in optimizer.shard_fp32_from_float16_groups
        for main_param in param_group
    )
    assert bind_gpu_staged_adamw(optimizer) == 1
    baseline = torch.optim.AdamW([baseline_param], **adam_kwargs)
    owned_shard = optimizer.optimizer.param_groups[0]["params"][0]

    for step in range(5):
        grad = torch.randn_like(model_param)
        model_param.main_grad = grad
        baseline_param.grad = grad.float()
        baseline.step()
        model_chunk.expected_param = baseline_param.detach().bfloat16()

        success, grad_norm, num_zeros = optimizer.step()
        inner.drain()

        assert success
        assert grad_norm == 0.0
        assert num_zeros == 0
        assert model_chunk.param_sync_count == step + 1
        staged_state = inner.state[owned_shard]
        baseline_state = baseline.state[baseline_param]
        torch.testing.assert_close(
            staged_state["master_param"],
            baseline_param.detach().cpu(),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            staged_state["exp_avg"],
            baseline_state["exp_avg"].cpu(),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            staged_state["exp_avg_sq"],
            baseline_state["exp_avg_sq"].cpu(),
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_gpu_state_residency_is_bounded_by_slots_not_total_state() -> None:
    """Larger owned state does not create additional resident CUDA master/moments."""
    optimizers = []
    observed = []
    for numel in (64, 1024):
        param = torch.nn.Parameter(
            torch.linspace(-1, 1, numel, device="cuda", dtype=torch.bfloat16)
        )
        optimizer = GPUStagedAdamW(
            [param], staged_config=_tiny_config(buffer_count=2, bucket_numel=16)
        )
        optimizer.bind_owned_params(optimizer.param_groups)
        param.decoupled_grad = torch.ones_like(param)
        optimizer.step()
        optimizer.drain()
        optimizers.append(optimizer)
        observed.append(optimizer.gpu_staging_state_numel)
        assert all(
            not tensor.is_cuda
            for state in optimizer.state.values()
            for tensor in state.values()
            if isinstance(tensor, torch.Tensor)
        )

    assert observed == [2 * 16 * 3, 2 * 16 * 3]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_checkpoint_state_dict_aliases_cpu_slabs_and_save_drains(monkeypatch) -> None:
    """Synchronous schema reads the authoritative pinned views without CUDA state."""
    param = torch.nn.Parameter(
        torch.linspace(-1, 1, 19, device="cuda", dtype=torch.bfloat16)
    )
    optimizer = GPUStagedAdamW(
        [param], staged_config=_tiny_config(buffer_count=1, bucket_numel=7)
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    param.decoupled_grad = torch.ones_like(param)
    optimizer.step()
    drain_calls = 0
    original_drain = optimizer.drain

    def tracked_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        original_drain()

    monkeypatch.setattr(optimizer, "drain", tracked_drain)
    wrapper = SimpleNamespace(optimizer=optimizer)
    prepare_managed_checkpoint_save(wrapper, async_save=False)
    state_dict = optimizer.state_dict()
    state = state_dict["state"][0]
    live_state = optimizer.state[param]

    assert drain_calls >= 2
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0
    assert set(state) == {"master_param", "exp_avg", "exp_avg_sq"}
    for key, tensor in state.items():
        assert tensor.device.type == "cpu"
        assert tensor.dtype is torch.float32
        assert tensor.is_pinned()
        assert (
            tensor.untyped_storage().data_ptr()
            == live_state[key].untyped_storage().data_ptr()
        )

    assert prepare_managed_checkpoint_save(wrapper, async_save=True) == (optimizer,)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_async_checkpoint_fence_waits_before_step_and_preserves_source() -> None:
    param = torch.nn.Parameter(torch.ones(19, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedAdamW(
        [param], staged_config=_tiny_config(buffer_count=1, bucket_numel=7)
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    wrapper = SimpleNamespace(optimizer=optimizer)
    waits = 0

    def wait_for_save() -> None:
        nonlocal waits
        waits += 1
        complete_managed_async_checkpoint_save((optimizer,))

    leaves = begin_managed_async_checkpoint_save(
        wrapper,
        checkpoint_id="checkpoint-1",
        path="/tmp/checkpoint-1",
        control_group=object(),
        wait_fn=wait_for_save,
        identities={(): {"leaf": "test"}},
    )
    bind_managed_async_checkpoint_request(leaves, object(), 0)
    checkpoint_master = optimizer.cpu_slabs.master.clone()
    param.decoupled_grad = torch.ones_like(param)

    optimizer.step()
    optimizer.drain()

    assert waits == 1
    assert optimizer.async_save_state == "COMPLETE"
    assert not torch.equal(optimizer.cpu_slabs.master, checkpoint_master)
    assert optimizer.cuda_state_numel == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_checkpoint_load_restores_cpu_state_and_continues_identically() -> None:
    """A fresh instance resumes model, master, moments, group step, and next update."""
    torch.manual_seed(20260814)
    initial = torch.randn(29, device="cuda", dtype=torch.bfloat16)
    kwargs = {"lr": 2e-3, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.03}
    source_param = torch.nn.Parameter(initial.clone())
    source = GPUStagedAdamW(
        [source_param], staged_config=_tiny_config(bucket_numel=8), **kwargs
    )
    source.bind_owned_params(source.param_groups)
    for _ in range(3):
        source_param.decoupled_grad = torch.randn_like(source_param)
        source.step()
        source.drain()
    checkpoint = source.state_dict()

    resumed_param = torch.nn.Parameter(source_param.detach().clone())
    resumed = GPUStagedAdamW(
        [resumed_param], staged_config=_tiny_config(bucket_numel=8), **kwargs
    )
    resumed.bind_owned_params(resumed.param_groups)
    managed = begin_managed_checkpoint_load(SimpleNamespace(optimizer=resumed))
    resumed.load_state_dict(checkpoint)
    complete_managed_checkpoint_load(managed)

    assert resumed.residency == "CPU_RESIDENT"
    assert resumed.cuda_state_numel == 0
    assert resumed.param_groups[0]["step"] == 3
    for key in ("master_param", "exp_avg", "exp_avg_sq"):
        actual = resumed.state[resumed_param][key]
        expected = source.state[source_param][key]
        assert actual.is_pinned()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    grad = torch.randn_like(source_param)
    source_param.decoupled_grad = grad
    resumed_param.decoupled_grad = grad.clone()
    source.step()
    resumed.step()
    source.drain()
    resumed.drain()
    torch.testing.assert_close(resumed_param, source_param, rtol=0.0, atol=0.0)
    for key in ("master_param", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            resumed.state[resumed_param][key],
            source.state[source_param][key],
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_checkpoint_load_failure_rolls_back_and_fails_closed() -> None:
    """A malformed state cannot leave partially loaded moments trainable."""
    param = torch.nn.Parameter(torch.ones(13, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedAdamW([param], staged_config=_tiny_config(bucket_numel=5))
    optimizer.bind_owned_params(optimizer.param_groups)
    before = {key: value.clone() for key, value in optimizer.state[param].items()}
    live_state_dict = optimizer.state_dict()
    malformed = {
        "state": {
            index: {key: value.clone() for key, value in state.items()}
            for index, state in live_state_dict["state"].items()
        },
        "param_groups": [dict(group) for group in live_state_dict["param_groups"]],
    }
    del malformed["state"][0]["exp_avg_sq"]
    managed = begin_managed_checkpoint_load(SimpleNamespace(optimizer=optimizer))

    with pytest.raises(KeyError, match="missing exp_avg_sq") as exc_info:
        optimizer.load_state_dict(malformed)
    abort_managed_checkpoint_load(managed, exc_info.value)

    for key, expected in before.items():
        torch.testing.assert_close(
            optimizer.state[param][key], expected, rtol=0.0, atol=0.0
        )
    param.decoupled_grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="failed checkpoint load"):
        optimizer.step()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for staged AdamW")
def test_model_only_recovery_streams_master_and_zeros_moments() -> None:
    """Missing optimizer state rebuilds CPU slabs from the loaded model shard."""
    param = torch.nn.Parameter(torch.zeros(23, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedAdamW([param], staged_config=_tiny_config(bucket_numel=6))
    optimizer.bind_owned_params(optimizer.param_groups)
    optimizer.cpu_slabs.exp_avg.fill_(4)
    optimizer.cpu_slabs.exp_avg_sq.fill_(9)
    optimizer.param_groups[0]["step"] = 17
    with torch.no_grad():
        param.copy_(torch.linspace(-2, 2, param.numel(), device="cuda"))

    assert reset_managed_optimizer_from_model(SimpleNamespace(optimizer=optimizer)) == 1

    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0
    assert optimizer.param_groups[0]["step"] == 0
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, param.detach().cpu().float(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg,
        torch.zeros_like(optimizer.cpu_slabs.exp_avg),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.exp_avg_sq,
        torch.zeros_like(optimizer.cpu_slabs.exp_avg_sq),
        rtol=0.0,
        atol=0.0,
    )


def test_bind_helper_leaves_non_managed_optimizer_unchanged() -> None:
    """Capability dispatch is a no-op for ordinary optimizers."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    wrapper = SimpleNamespace(optimizer=optimizer)
    original_groups = optimizer.param_groups

    assert bind_gpu_staged_adamw(wrapper) == 0
    assert optimizer.param_groups is original_groups
    param.grad = torch.tensor([0.25])
    optimizer.step()
    assert param.item() != 1.0


def _compatible_mcore_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        use_distributed_optimizer=True,
        bf16=True,
        optimizer_cpu_offload=False,
        use_precision_aware_optimizer=True,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        optimizer_cuda_graph=False,
        fp8_recipe=None,
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.01,
        decoupled_weight_decay=True,
    )


def test_builder_concurrency_never_mutates_global_adam(monkeypatch) -> None:
    """Concurrent staged/staged and staged/ordinary builds leave Adam untouched."""
    import megatron.core.optimizer as mcore_optimizer

    from areal.engine.megatron_utils import gpu_staged_optimizer as staged_module

    original_adam = mcore_optimizer.Adam

    def run_pair(staged_flags: tuple[bool, bool]) -> None:
        barrier = threading.Barrier(2)

        def fake_builder(config, model):
            del config, model
            barrier.wait(timeout=5)
            return object()

        monkeypatch.setattr(mcore_optimizer, "get_megatron_optimizer", fake_builder)
        monkeypatch.setattr(
            staged_module,
            "_replace_metadata_optimizers_with_staged_adamw",
            lambda optimizer, mcore_config, staged_config: 1,
        )
        monkeypatch.setattr(staged_module, "bind_gpu_staged_adamw", lambda _: 1)

        def build(staged: bool):
            if staged:
                return get_megatron_optimizer_with_gpu_staged_adamw(
                    _compatible_mcore_config(),
                    [object()],
                    GPUStagedAdamWConfig(buffer_count=1, bucket_size_mb=1),
                )
            return mcore_optimizer.get_megatron_optimizer(
                _compatible_mcore_config(), [object()]
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(build, staged_flags))
        assert len(results) == 2
        assert mcore_optimizer.Adam is original_adam

    run_pair((True, True))
    run_pair((True, False))
    run_pair((False, False))
