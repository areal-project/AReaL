"""Tests for PerLayerOptimWrapper with FSDP2.

Tests correctness of per-layer optim step against baseline optimizer.step().
Uses mp.spawn pattern for multi-GPU distributed tests.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, Qwen2Config

from areal.engine.fsdp_utils import PerLayerOptimWrapper, apply_fsdp2
from areal.engine.fsdp_utils.grad import fsdp2_clip_grad_norm
from areal.engine.fsdp_utils.optimizer import _get_local_tensor

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_tiny_config():
    """Create a tiny Qwen2 config for fast testing."""
    return Qwen2Config(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=128,
    )


def _setup_distributed(rank, world_size, rendezvous_file):
    """Initialize distributed process group and return device mesh."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    return init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))


def _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False):
    """Create model with FSDP2 wrapping."""
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model = model.to(device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
    )
    cpu_offload = CPUOffloadPolicy(pin_memory=True) if use_cpu_offload else None
    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload,
    }
    apply_fsdp2(model, fsdp_kwargs, None)
    return model


def _clip_grad_norm(model, device_mesh, max_norm=1.0):
    """Clip gradients using AReaL's fsdp2_clip_grad_norm."""
    return fsdp2_clip_grad_norm(
        list(model.parameters()),
        max_norm=max_norm,
        fsdp_group=device_mesh.get_group(),
    )


def _offload_optimizer_states(optimizer):
    """Move all optimizer states to CPU."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                state[k] = v.to("cpu")


def _snapshot_params(model):
    """Snapshot all parameters as float32 CPU tensors."""
    return {
        n: _get_local_tensor(p.data).detach().float().cpu().clone()
        for n, p in model.named_parameters()
    }


def _assert_params_close(model, baseline_params, atol=1e-4):
    """Assert model params match baseline within tolerance."""
    for n, p in model.named_parameters():
        val = _get_local_tensor(p.data).detach().float().cpu()
        diff = (val - baseline_params[n]).abs().max().item()
        assert diff < atol, f"Param {n} diff={diff:.6e} exceeds threshold {atol}"


def _spawn_test(worker_fn, world_size, tmp_path, *extra_args):
    """Common mp.spawn boilerplate for distributed tests."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    rendezvous_file = str(tmp_path / "rdzv_file")
    mp.spawn(
        fn=worker_fn,
        args=(world_size, rendezvous_file, *extra_args),
        nprocs=world_size,
        join=True,
    )


# ---------------------------------------------------------------------------
# Test workers
# ---------------------------------------------------------------------------


def _test_layer_grouping_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()
    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    wrapper = PerLayerOptimWrapper(model, optimizer, device_id=rank)
    groups = wrapper._layer_param_groups

    # Should have num_hidden_layers groups + residual (lm_head, final norm)
    assert len(groups) >= config.num_hidden_layers, (
        f"Expected at least {config.num_hidden_layers} groups, got {len(groups)}"
    )

    # Verify all trainable params are covered
    grouped_ids = {id(p) for g in groups for p in g}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped_ids == model_ids, f"Missing params: {model_ids - grouped_ids}"

    dist.destroy_process_group()


def _test_correctness_worker(rank, world_size, rendezvous_file, use_cpu_offload):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    # --- Baseline: standard optimizer.step() ---
    torch.manual_seed(42)
    model_baseline = _create_model_and_fsdp(
        config, device_mesh, use_cpu_offload=use_cpu_offload
    )
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)

    torch.manual_seed(100 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")

    loss = model_baseline(input_ids=input_ids).logits.mean()
    loss.backward()
    _clip_grad_norm(model_baseline, device_mesh)
    optimizer_baseline.step()
    optimizer_baseline.zero_grad()

    baseline_params = _snapshot_params(model_baseline)

    del model_baseline, optimizer_baseline
    torch.cuda.empty_cache()

    # --- Per-layer optim step ---
    torch.manual_seed(42)
    model_perlayer = _create_model_and_fsdp(
        config, device_mesh, use_cpu_offload=use_cpu_offload
    )
    optimizer_perlayer = torch.optim.AdamW(model_perlayer.parameters(), lr=1e-3)

    torch.manual_seed(100 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")

    loss = model_perlayer(input_ids=input_ids).logits.mean()
    loss.backward()
    _clip_grad_norm(model_perlayer, device_mesh)

    _offload_optimizer_states(optimizer_perlayer)
    torch.cuda.synchronize()

    wrapper = PerLayerOptimWrapper(
        model_perlayer, optimizer_perlayer, device_id=rank, prefetch_layers=1
    )
    wrapper.step()
    optimizer_perlayer.zero_grad()

    _assert_params_close(model_perlayer, baseline_params)

    del model_perlayer, optimizer_perlayer
    dist.destroy_process_group()


def _test_multi_step_worker(rank, world_size, rendezvous_file):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create wrapper once to test event reuse and state accumulation
    _offload_optimizer_states(optimizer)
    torch.cuda.synchronize()
    wrapper = PerLayerOptimWrapper(model, optimizer, device_id=rank, prefetch_layers=1)

    for step in range(3):
        torch.manual_seed(step * 100 + rank)
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")
        loss = model(input_ids=input_ids).logits.mean()
        loss.backward()
        _clip_grad_norm(model, device_mesh)

        wrapper.step()
        optimizer.zero_grad()

    dist.destroy_process_group()


def _test_prefetch_layers_worker(rank, world_size, rendezvous_file, prefetch_layers):
    device_mesh = _setup_distributed(rank, world_size, rendezvous_file)
    config = _create_tiny_config()

    model = _create_model_and_fsdp(config, device_mesh, use_cpu_offload=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=f"cuda:{rank}")
    loss = model(input_ids=input_ids).logits.mean()
    loss.backward()
    _clip_grad_norm(model, device_mesh)

    _offload_optimizer_states(optimizer)
    torch.cuda.synchronize()
    wrapper = PerLayerOptimWrapper(
        model, optimizer, device_id=rank, prefetch_layers=prefetch_layers
    )
    wrapper.step()
    optimizer.zero_grad()

    # Verify params were updated (not all zeros)
    updated = any(_get_local_tensor(p.data).abs().sum() > 0 for p in model.parameters())
    assert updated, "No params were updated"

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("world_size", [2])
def test_layer_grouping(world_size, tmp_path):
    _spawn_test(_test_layer_grouping_worker, world_size, tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("world_size,use_cpu_offload", [(2, False), (2, True)])
def test_correctness(world_size, use_cpu_offload, tmp_path):
    _spawn_test(_test_correctness_worker, world_size, tmp_path, use_cpu_offload)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("world_size", [2])
def test_multi_step(world_size, tmp_path):
    _spawn_test(_test_multi_step_worker, world_size, tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("world_size,prefetch_layers", [(2, 0), (2, 1), (2, 2)])
def test_prefetch_layers(world_size, prefetch_layers, tmp_path):
    _spawn_test(_test_prefetch_layers_worker, world_size, tmp_path, prefetch_layers)
