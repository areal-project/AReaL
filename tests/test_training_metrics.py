# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for model work estimates and distributed metric aggregation."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint

from areal.utils.flops import get_flops_estimator, register_flops_estimator
from areal.utils.moe_metrics import MoEMetrics, normalize_expert_loads
from areal.utils.training_metrics import TrainingMetrics


def _config(hybrid=False):
    return SimpleNamespace(
        model_type="qwen3_5_moe_text" if hybrid else "qwen3_moe",
        hidden_size=2048,
        num_hidden_layers=40 if hybrid else 48,
        num_attention_heads=16 if hybrid else 32,
        num_key_value_heads=2 if hybrid else 4,
        head_dim=256 if hybrid else 128,
        num_experts=256 if hybrid else 128,
        num_experts_per_tok=8,
        moe_intermediate_size=512 if hybrid else 768,
        shared_expert_intermediate_size=512 if hybrid else 0,
        vocab_size=248320 if hybrid else 151936,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_flops_sequence_lengths_preserve_quadratic_attention(hybrid):
    """Packing must sum individual FLOPs, never square the packed token count."""
    config = _config(hybrid)
    config.layer_types *= 10
    estimate = get_flops_estimator(config)
    full_layers = 10 if hybrid else 48
    coefficient = 3 * full_layers * 2 * config.num_attention_heads * config.head_dim
    assert estimate(0) == 0
    assert estimate(20) - 2 * estimate(10) == coefficient * 200
    assert estimate(10) > 0
    with pytest.raises(ValueError, match="nonnegative integer"):
        estimate(-1)


def test_flops_tiny_qwen_matches_hand_counted_projections():
    """Count Q/K/V/O, selected MLPs, router, head and causal attention by hand."""
    config = SimpleNamespace(
        model_type="qwen3_moe",
        hidden_size=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        vocab_size=5,
    )
    # Per-token: attention projections=24, MLP=72, router=16, LM head=20.
    # Length 3: 6 causal pairs * 2 heads * 1 dim * 4 FLOPs = 48.
    assert get_flops_estimator(config)(3) == 3 * (3 * 132 + 48)


def test_flops_hybrid_all_linear_has_no_quadratic_term():
    """DeltaNet state work grows linearly with sequence length."""
    config = _config(True)
    config.layer_types = ["linear_attention"] * config.num_hidden_layers
    estimate = get_flops_estimator(SimpleNamespace(text_config=config))
    assert estimate(200) == 2 * estimate(100)


def test_flops_custom_registry_and_unknown_model():
    """Custom estimators receive the actual checkpoint configuration."""
    register_flops_estimator("test_custom", lambda config: lambda n: config.factor * n)
    config = SimpleNamespace(model_type="test_custom", factor=42)
    assert get_flops_estimator(config)(10) == 420
    assert get_flops_estimator(SimpleNamespace(model_type="unknown")) is None


def test_training_metrics_interval_and_cumulative_are_ratios_of_sums(monkeypatch):
    """Exclude masked padding and aggregate unequal-duration optimizer calls."""
    metrics = TrainingMetrics(lambda n: n * n)
    times = iter([0.0, 2.0, 3.0, 7.0, 10.0, 12.0])
    monkeypatch.setattr(
        "areal.utils.training_metrics.time.perf_counter", lambda: next(times)
    )
    batch = {"attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    synchronizations = []
    for _ in range(2):
        with metrics.measure(batch, lambda: synchronizations.append(True)):
            pass
    result = metrics.export(dp_group=None, timing_group=None)
    assert len(synchronizations) == 4
    assert result["train_perf/tokens"] == 10
    assert result["train_perf/tokens_per_second"] == pytest.approx(10 / 6)
    assert result["train_perf/estimated_flops_per_second"] == pytest.approx(26 / 6)
    assert metrics.export(dp_group=None, timing_group=None) == {}
    with metrics.measure(batch, lambda: None):
        pass
    result = metrics.export(dp_group=None, timing_group=None)
    assert result["train_perf/cumulative_tokens_per_second"] == 15 / 8


def test_training_metrics_unknown_model_omits_flops():
    """Unsupported architectures still provide token throughput."""
    metrics = TrainingMetrics(None)
    metrics._pending = [(torch.tensor([4]), 2.0)]
    result = metrics.export(dp_group=None, timing_group=None)
    assert result["train_perf/tokens_per_second"] == 2
    assert all("flops" not in key for key in result)


@pytest.mark.parametrize(
    "counts,percent,ratio",
    [
        ([2, 2], [50, 50], 1),
        ([3, 1], [75, 25], 1.5),
        ([4, 0], [100, 0], 2),
        ([0, 0], [0, 0], 0),
    ],
)
def test_moe_normalization_load_cases_match_expected(counts, percent, ratio):
    """Normalize assignments, with a defined empty-layer result."""
    actual, imbalance = normalize_expert_loads(torch.tensor(counts))
    torch.testing.assert_close(
        actual, torch.tensor(percent, dtype=torch.float64), rtol=0, atol=0
    )
    assert imbalance.item() == ratio


class _Router(nn.Module):
    def forward(self, x):
        return x.sin(), None, torch.tensor([3, 1], device=x.device)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.register_buffer("routing_counts", torch.zeros(2, dtype=torch.int64))

    def forward(self, x):
        y, _, counts = self.router(x)
        self.routing_counts.copy_(counts)
        return y.cos()


class _Model(nn.Module):
    def __init__(self, block, recompute):
        super().__init__()
        self.block = block
        self.recompute = recompute

    def forward(self, x):
        if self.recompute:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


@pytest.mark.parametrize("buffers", [False, True])
@pytest.mark.parametrize("as_iterator", [False, True])
def test_moe_collector_empty_modules_registers_no_hooks(buffers, as_iterator):
    """Dense model parts must not pay per-forward Python hook overhead."""
    model = nn.Identity()
    metrics = MoEMetrics()
    modules = iter(()) if as_iterator else []
    if buffers:
        metrics.attach_buffers(model, modules)
    else:
        metrics.attach(model, modules, lambda output: output)
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    with metrics.measure():
        model(torch.ones(2))
    assert metrics.export(reduce_group=None) == {}
    metrics.close()


@pytest.mark.parametrize("buffers", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_moe_collector_excludes_eval_and_backward_recomputation(buffers, recompute):
    """Original model forwards count once even when inner layers are replayed."""
    block = _Block()
    model = _Model(block, recompute)
    metrics = MoEMetrics()
    if buffers:
        metrics.attach_buffers(model, [("7", block)])
    else:
        metrics.attach(model, [("7", block.router)], lambda output: output[2])
    x = torch.randn(4, requires_grad=True)
    model(x)  # evaluation/reference forward outside training
    with metrics.measure():
        model(x).sum().backward()
        model(x).sum().backward()
    result = metrics.export(reduce_group=None)
    assert result["moe_balance/layer_7/expert_0/tokens"] == 6
    assert result["moe_balance/layer_7/expert_0/load_percent"] == 75
    assert result["moe_balance/layer_7/max_over_ideal"] == 1.5
    assert metrics.export(reduce_group=None) == {}
    metrics.close()


@pytest.mark.slow
@pytest.mark.ci
def test_moe_buffer_collection_with_fullgraph_compile_counts_once():
    """Archon's outer hook permits fullgraph compilation of the inner layer."""
    block = _Block()
    model = _Model(torch.compile(block, backend="aot_eager", fullgraph=True), True)
    metrics = MoEMetrics()
    metrics.attach_buffers(model, [("0", block)])
    with metrics.measure():
        model(torch.randn(4, requires_grad=True)).sum().backward()
    result = metrics.export(reduce_group=None)
    assert result["moe_balance/layer_0/expert_0/tokens"] == 3
    metrics.close()


def _distributed_worker(rank, rendezvous, backend="gloo"):
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend, init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    world = dist.new_group([0, 1], backend=backend)
    timing = dist.new_group([0, 1], backend="gloo")
    local_groups = [dist.new_group([i], backend=backend) for i in range(2)]
    try:
        metrics = TrainingMetrics(lambda n: n * n)
        metrics._pending = [(torch.tensor([3 + 4 * rank], device=device), 2 + 2 * rank)]
        result = metrics.export(dp_group=world, timing_group=timing)
        assert result["train_perf/tokens"] == 10
        assert result["train_perf/tokens_per_second"] == 2.5
        assert result["train_perf/estimated_flops_per_second"] == 58 / 4
        # Two model-parallel replicas: each has the same logical input.
        metrics = TrainingMetrics(None)
        metrics._pending = [(torch.tensor([3], device=device), 2 + 2 * rank)]
        result = metrics.export(dp_group=local_groups[rank], timing_group=timing)
        assert result["train_perf/tokens_per_second"] == 3 / 4
        # Unequal local loads must be summed BEFORE normalization.
        moe = MoEMetrics()
        moe.counts["0"] = (
            torch.tensor([3, 1], device=device)
            if rank == 0
            else torch.tensor([1, 5], device=device)
        )
        result = moe.export(reduce_group=world)
        assert result["moe_balance/layer_0/expert_0/load_percent"] == 40
        assert result["moe_balance/layer_0/max_over_ideal"] == 1.2
        # Different PP stages own distinct global layer IDs.
        moe.counts[str(rank)] = torch.tensor([3, 1], device=device)
        result = moe.export(reduce_group=local_groups[rank], pp_group=world)
        assert result["moe_balance/layer_0/max_over_ideal"] == 1.5
        assert result["moe_balance/layer_1/max_over_ideal"] == 1.5
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_distributed_work_and_moe_reductions_preserve_topology(tmp_path):
    """Two real CPU/Gloo ranks test SUM/MAX, replica exclusion, and PP union."""
    mp.spawn(
        _distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True
    )


def test_expert_load_table_preserves_ids_counts_and_scalar_ratios():
    """W&B receives one expert matrix table and the per-layer ratio scalars."""
    from areal.utils.moe_metrics import split_expert_load_metrics

    scalar, rows = split_expert_load_metrics(
        {
            "moe_balance/layer_10/expert_2/tokens": 7,
            "moe_balance/layer_10/expert_2/load_percent": 70,
            "moe_balance/layer_2/expert_1/tokens": 3,
            "moe_balance/layer_2/expert_1/load_percent": 30,
            "moe_balance/layer_10/max_over_ideal": 1.4,
            "train_perf/tokens_per_second": 100,
        }
    )
    assert rows == [[2, 1, 3, 30], [10, 2, 7, 70]]
    assert scalar == {
        "moe_balance/layer_10/max_over_ideal": 1.4,
        "train_perf/tokens_per_second": 100,
    }


def test_stats_logger_logs_expert_table_without_expert_scalar_series(monkeypatch):
    """Exercise W&B payload construction without creating or publishing a run."""
    from areal.utils import stats_logger

    logged = []
    monkeypatch.setattr(
        stats_logger.wandb, "log", lambda data, step: logged.append((data, step))
    )
    monkeypatch.setattr(stats_logger.swanlab, "log", lambda *args, **kwargs: None)
    logger = stats_logger.StatsLogger.__new__(stats_logger.StatsLogger)
    logger.ft_spec = SimpleNamespace(
        total_train_epochs=1, steps_per_epoch=1, total_train_steps=1
    )
    logger._last_commit_step = -1
    tensorboard_logged = []
    logger.summary_writer = SimpleNamespace(
        add_scalar=lambda key, value, step: tensorboard_logged.append(
            (key, value, step)
        )
    )
    logger.print_stats = lambda data: None
    logger.commit(
        0,
        0,
        0,
        {
            "moe_balance/layer_0/expert_0/tokens": 3,
            "moe_balance/layer_0/expert_0/load_percent": 75,
            "moe_balance/layer_0/max_over_ideal": 1.5,
            "train_perf/tokens_per_second": 123,
        },
    )
    payload, step = logged[0]
    assert step == 0
    assert payload["moe_balance/expert_loads"].columns == [
        "layer",
        "expert",
        "tokens",
        "load_percent",
    ]
    assert payload["moe_balance/expert_loads"].data == [[0, 0, 3, 75]]
    assert payload["moe_balance/layer_0/max_over_ideal"] == 1.5
    assert payload["train_perf/tokens_per_second"] == 123
    assert not any("/expert_0/" in key for key in payload)

    assert tensorboard_logged == [
        ("moe_balance/layer_0/max_over_ideal", 1.5, 0),
        ("train_perf/tokens_per_second", 123, 0),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Archon MoE kernels"
)
@pytest.mark.parametrize("recompute", [False, True])
def test_archon_moe_cuda_buffer_counts_original_training_forward(recompute):
    """Validate the real MoE routing buffer with Triton routing and backward."""
    pytest.importorskip("triton")
    from areal.experimental.models.archon.moe import MoE, MoEArgs

    moe = MoE(
        MoEArgs(
            num_experts=4, top_k=2, use_grouped_mm=False, _debug_force_load_balance=True
        ),
        dim=16,
        hidden_dim=32,
    ).to(device="cuda", dtype=torch.bfloat16)
    moe.init_buffers("cuda")
    moe.init_weights(0.02)
    model = _Model(moe, recompute)
    metrics = MoEMetrics()
    metrics.attach_buffers(model, [("0", moe)])
    x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    model(x)  # excluded evaluation/reference computation
    with metrics.measure():
        model(x).float().sum().backward()
    result = metrics.export(reduce_group=None)
    assert sum(result[f"moe_balance/layer_0/expert_{e}/tokens"] for e in range(4)) == 16
    assert (
        sum(result[f"moe_balance/layer_0/expert_{e}/load_percent"] for e in range(4))
        == 100
    )
    assert result["moe_balance/layer_0/max_over_ideal"] == 1
    assert torch.isfinite(x.grad).all()
    assert "routing_counts" not in moe.state_dict()
    metrics.close()


@pytest.mark.slow
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two CUDA devices required for NCCL"
)
def test_distributed_nccl_metrics_reduce_cuda_work_and_cpu_timing(tmp_path):
    """Validate actual NCCL work/load collectives alongside the Gloo time group."""
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "nccl_rendezvous"), "nccl"),
        nprocs=2,
        join=True,
    )


def test_stats_logger_large_expert_table_serializes_every_layer(monkeypatch):
    """W&B run-media serialization must retain Qwen3.5's 10,240 experts."""
    from areal.utils import stats_logger

    monkeypatch.setattr(stats_logger.wandb.Table, "MAX_ROWS", 10000)
    serialized = []
    monkeypatch.setattr(
        stats_logger.wandb,
        "log",
        lambda data, step: serialized.append(
            data["moe_balance/expert_loads"]._to_table_json()
        ),
    )
    monkeypatch.setattr(stats_logger.swanlab, "log", lambda *args, **kwargs: None)
    logger = stats_logger.StatsLogger.__new__(stats_logger.StatsLogger)
    logger.ft_spec = SimpleNamespace(
        total_train_epochs=1, steps_per_epoch=1, total_train_steps=1
    )
    logger._last_commit_step = -1
    logger.summary_writer = None
    logger.print_stats = lambda data: None
    metrics = {}
    for layer in range(40):
        for expert in range(256):
            prefix = f"moe_balance/layer_{layer}/expert_{expert}"
            metrics[prefix + "/tokens"] = 1
            metrics[prefix + "/load_percent"] = 100 / 256
    logger.commit(0, 0, 0, metrics)
    assert len(serialized[0]["data"]) == 10240
    assert serialized[0]["data"][-1] == [39, 255, 1, 100 / 256]
