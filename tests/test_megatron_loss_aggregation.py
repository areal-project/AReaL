# SPDX-License-Identifier: Apache-2.0

"""Independent loss oracles and callback tests; real collectives run via torchrun."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function
from tests.test_megatron_loss_normalization import (
    ENGINE_SOURCE,
    _bind_per_token_loss_methods,
    _MicroBatches,
)

from areal.infra.platforms import current_platform
from areal.trainer.ppo.actor import trajectory_token_weights
from areal.utils.functional.loss_aggregation import PolicyGradientReduction

MEAN_MODES = ("token-mean", "seq-mean", "traj-mean", "prompt-mean")
# Two prompts, two trajectories per prompt. Include empty rows inside trajectories.
TRAJECTORY_IDS = (0, 0, 0, 1, 2, 3, 3, 3)
PROMPT_IDS = (0, 0, 0, 0, 1, 1, 1, 1)


def make_loss_data(mode, scenario="ragged", device="cpu"):
    """Keep prompt tokens, tool holes and padding in the physical token layout."""
    mask = torch.zeros(8, 16, dtype=torch.bool, device=device)
    for row, length in enumerate((1, 3, 0, 2, 2, 1, 4, 0)):
        mask[row, 1 : 1 + 2 * length : 2] = True
    if scenario == "empty-unit":
        mask[4] = False
    elif scenario == "empty-prompt":
        mask[4:] = False
    elif scenario == "empty-rank":
        mask[1::2] = False
    numerator_mask = mask.clone()
    if scenario == "filtered":
        numerator_mask[:4] = False
        numerator_mask[6, 3] = False
    starts = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0], device=device)
    return {
        "targets": torch.linspace(-0.4, 0.8, 128, device=device).reshape(8, 16),
        "loss_mask": mask,
        "numerator_mask": numerator_mask,
        "unit_weights": trajectory_token_weights(
            mask, starts, 2 if mode == "prompt-mean" else 1
        ),
    }


def independent_loss_oracle(values, data, mode):
    """Derive the objective from explicit logical IDs, without production reducers."""
    mask, kept = data["loss_mask"], data["numerator_mask"]
    if mode == "token-mean":
        return torch.where(kept, values, 0).sum() / mask.count_nonzero()
    ids = {
        "seq-mean": tuple(range(8)),
        "traj-mean": TRAJECTORY_IDS,
        "prompt-mean": PROMPT_IDS,
    }[mode]
    means = []
    for unit in sorted(set(ids)):
        rows = [row for row, owner in enumerate(ids) if owner == unit]
        count = mask[rows].count_nonzero()
        if count > 0:
            means.append(torch.where(kept[rows], values[rows], 0).sum() / count)
    return torch.stack(means).mean()


def make_loss_microbatches(data, dp_rank, dp_size, split, packed=True):
    """Split logical units across DP and forward batches after preparing weights."""
    rows = list(range(dp_rank, 8, dp_size))
    partitions = [[row] for row in rows] if split else [rows]
    batches = _MicroBatches()
    for partition in partitions:
        batch = {key: value[partition] for key, value in data.items()}
        batch["rows"] = partition
        if packed:
            batch = {
                key: value.flatten() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch["cu_seqlens"] = (
                torch.arange(
                    len(partition) + 1,
                    dtype=torch.int32,
                    device=data["loss_mask"].device,
                )
                * 16
            )
        batches.append(batch)
    return batches


def aggregate_test_loss(values, batch, reduction):
    return reduction.aggregate(
        (values - batch["targets"]).square(),
        batch["numerator_mask"],
        denominator_mask=batch["loss_mask"],
        cu_seqlens=batch.get("cu_seqlens"),
        unit_weights=batch["unit_weights"],
    )


def load_engine_callbacks(namespace):
    return tuple(
        _load_function(ENGINE_SOURCE, name, namespace)
        for name in (
            "train_batch",
            "_compute_logprobs_and_loss",
            "_build_per_token_loss_output",
        )
    )


def run_callback_contract(mode, cp_size, dp_size, split, per_token, scenario):
    """Model the scalar schedule contract; do not claim to emulate CP collectives."""
    data = make_loss_data(mode, scenario)
    parameter = torch.linspace(-0.2, 0.6, 128, dtype=torch.float64).reshape(8, 16)
    parameter.requires_grad_()
    reduction = PolicyGradientReduction(mode)
    batches = [
        make_loss_microbatches(data, rank, dp_size, split) for rank in range(dp_size)
    ]
    losses, counts = [], []

    def global_weight(mbs, fn, group):
        total = sum(fn(mb) for rank_batches in batches for mb in rank_batches)
        return total * (cp_size if group == "dp_cp" else 1)

    def all_reduce(totals, group):
        totals.copy_(
            torch.stack(
                (
                    global_weight(None, reduction.normalizer_fn, group),
                    global_weight(
                        None, lambda mb: mb["loss_mask"].count_nonzero(), group
                    ),
                )
            ).to(totals)
        )

    for rank_batches in batches:
        for cp_rank in range(cp_size):
            namespace = {
                "torch": torch,
                "dist": SimpleNamespace(all_reduce=all_reduce),
                "get_model_config": lambda model: model.config,
                "compute_total_loss_weight": global_weight,
                "mpu": SimpleNamespace(
                    get_data_parallel_group=lambda with_context_parallel=False: (
                        "dp_cp" if with_context_parallel else "dp"
                    ),
                    get_data_parallel_world_size=lambda: dp_size,
                    get_context_parallel_world_size=lambda: cp_size,
                    get_context_parallel_rank=lambda: cp_rank,
                ),
            }
            train, compute, build = load_engine_callbacks(namespace)
            engine = SimpleNamespace(
                _ensure_ready=lambda: None,
                optimizer_zero_grad=lambda: None,
                optimizer_step=lambda: {},
                _normalize_batch_input=lambda inputs: (inputs, None),
                _prepare_mb_list=lambda inputs: rank_batches,
                model=[
                    SimpleNamespace(
                        config=SimpleNamespace(calculate_per_token_loss=per_token)
                    )
                ],
                config=SimpleNamespace(is_critic=True, loss_aggregation=mode),
                enable_tree_training=False,
                device="cpu",
            )
            engine._compute_logprobs_and_loss = lambda *a, **kw: compute(
                engine, *a, **kw
            )
            engine._build_per_token_loss_output = lambda *a, **kw: build(
                engine, *a, **kw
            )
            _bind_per_token_loss_methods(engine, namespace)

            def forward_backward(mbs, process_output, forward_only):
                assert not forward_only
                for batch in mbs:
                    output = parameter[batch["rows"]].reshape(-1, 1)
                    result = process_output(output, batch)
                    if per_token:
                        numerator, count = result
                        losses.append(numerator)
                        counts.append(count)
                    else:
                        # Two-value MCore schedule and DDP average contract.
                        losses.append(result / (len(mbs) * dp_size))

            engine.forward_backward_batch = forward_backward
            train(
                engine,
                {},
                lambda values, batch: aggregate_test_loss(values, batch, reduction),
                reduction.normalizer_fn,
            )
    actual = sum(losses) / sum(counts) if per_token else sum(losses)
    expected = independent_loss_oracle(
        (parameter - data["targets"]).square(), data, mode
    )
    return actual, expected, parameter, sum(counts), data


@pytest.mark.parametrize("mode", MEAN_MODES)
@pytest.mark.parametrize("cp_size", [1, 2, 8])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("split", [False, True], ids=["one-mb", "row-mbs"])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("scenario", ["ragged", "empty-unit", "filtered", "empty-rank"])
def test_mean_callback_matches_independent_token_gradients(
    mode, cp_size, dp_size, split, per_token, scenario
):
    """Unequal units retain their original coefficients across both split axes."""
    actual, expected, parameter, _, _ = run_callback_contract(
        mode, cp_size, dp_size, split, per_token, scenario
    )
    actual_grad = torch.autograd.grad(actual, parameter)[0]
    expected_grad = torch.autograd.grad(expected, parameter)[0]
    # Non-token reducers intentionally accumulate in float32; the oracle is fp64.
    torch.testing.assert_close(
        actual.to(expected.dtype), expected, rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("mode", MEAN_MODES)
@pytest.mark.parametrize("cp_size", [1, 2, 8])
def test_per_token_count_preserves_auxiliary_token_denominator(mode, cp_size):
    """Changing the main reduction must not change MCore's auxiliary divisor."""
    _, _, _, reported, data = run_callback_contract(
        mode, cp_size, dp_size=2, split=True, per_token=True, scenario="ragged"
    )
    torch.testing.assert_close(
        reported, data["loss_mask"].count_nonzero(), rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", MEAN_MODES)
@pytest.mark.parametrize("cp_size", [1, 2])
def test_originally_empty_prompt_is_excluded_from_mean(mode, cp_size):
    """An originally empty group differs from a nonempty group filtered later."""
    actual, expected, parameter, _, _ = run_callback_contract(
        mode, cp_size, dp_size=2, split=True, per_token=True, scenario="empty-prompt"
    )
    torch.testing.assert_close(
        actual.to(expected.dtype), expected, rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter)[0],
        torch.autograd.grad(expected, parameter)[0],
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", MEAN_MODES)
@pytest.mark.parametrize("scenario", ["ragged", "empty-unit", "filtered"])
def test_padded_and_packed_mean_match_independent_oracle(mode, scenario):
    """Packing retains prompt/tool holes and original denominators after filtering."""
    data = make_loss_data(mode, scenario)
    reduction = PolicyGradientReduction(mode)
    values = torch.full((8, 16), 0.07, requires_grad=True)
    expected = independent_loss_oracle((values - data["targets"]).square(), data, mode)
    gradients = []
    for packed in (False, True):
        batch = make_loss_microbatches(data, 0, 1, False, packed=packed)[0]
        actual = aggregate_test_loss(
            values.flatten() if packed else values, batch, reduction
        )
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
        gradients.append(torch.autograd.grad(actual, values)[0])
    expected_grad = torch.autograd.grad(expected, values)[0]
    for gradient in gradients:
        torch.testing.assert_close(gradient, expected_grad, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("mode", ["traj-mean", "prompt-mean"])
@pytest.mark.parametrize("prompt_count", [2, 4])
def test_eight_rollout_groups_with_fanout_survive_dp_and_forward_splits(
    mode, prompt_count
):
    """Exercise the original batch-size 2/4, group-size 8 examples with multi-row rollouts."""
    row_owners, starts = [], []
    for trajectory in range(prompt_count * 8):
        fanout = 1 + trajectory % 3
        row_owners.extend([trajectory] * fanout)
        starts.extend([1] + [0] * (fanout - 1))
    mask = torch.zeros(len(row_owners), 11, dtype=torch.bool)
    for row in range(len(row_owners)):
        mask[row, 1 : 2 + row % 9] = True
    values = torch.arange(mask.numel(), dtype=torch.float32).reshape_as(mask) / 100
    weights = trajectory_token_weights(
        mask, torch.tensor(starts), 8 if mode == "prompt-mean" else 1
    )
    reduction = PolicyGradientReduction(mode)
    owner_ids = [owner // 8 if mode == "prompt-mean" else owner for owner in row_owners]
    unit_means = []
    for unit in sorted(set(owner_ids)):
        rows = [row for row, owner in enumerate(owner_ids) if owner == unit]
        unit_means.append(values[rows][mask[rows]].mean())
    expected = torch.stack(unit_means).mean()
    numerator, denominator = 0.0, 0.0
    # Round-robin DP assignment and one-row forward batches cut both logical units.
    for dp_rank in range(2):
        for row in range(dp_rank, len(row_owners), 2):
            batch = {
                "loss_mask": mask[row : row + 1],
                "unit_weights": weights[row : row + 1],
            }
            weight = reduction.normalizer_fn(batch)
            local_mean = reduction.aggregate(
                values[row : row + 1],
                batch["loss_mask"],
                unit_weights=batch["unit_weights"],
            )
            numerator = numerator + local_mean * weight
            denominator = denominator + weight
    torch.testing.assert_close(numerator / denominator, expected, rtol=2e-6, atol=1e-7)


def test_missing_rollout_in_eight_sample_groups_is_not_silently_regrouped():
    """An absent {} rollout leaves 15 trajectories: traj-mean works, prompt-mean rejects."""
    starts = torch.tensor(
        [flag for trajectory in range(15) for flag in [1, 0][: 1 + trajectory % 2]]
    )
    mask = torch.ones(starts.numel(), 3, dtype=torch.bool)
    weights = trajectory_token_weights(mask, starts)
    torch.testing.assert_close(weights.sum(), torch.tensor(15.0), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="do not divide into groups"):
        trajectory_token_weights(mask, starts, 8)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dp_size,cp_size",
    [
        pytest.param(1, 1, marks=[pytest.mark.gpu, pytest.mark.npu]),
        pytest.param(2, 1, marks=[pytest.mark.multi_gpu, pytest.mark.multi_npu]),
        pytest.param(1, 2, marks=[pytest.mark.multi_gpu, pytest.mark.multi_npu]),
        pytest.param(2, 2, marks=[pytest.mark.multi_gpu, pytest.mark.multi_npu]),
    ],
)
@pytest.mark.parametrize("check", ["main", "auxiliary"])
@pytest.mark.parametrize("prepared", [False, True], ids=["legacy", "update-protocol"])
def test_real_mcore_loss_aggregation_gradients(dp_size, cp_size, check, prepared):
    """Exercise actual dependency DDP/schedule/finalize and differentiable CP gather."""
    if current_platform.device_type not in ("cuda", "npu"):
        pytest.skip("Requires CUDA or Ascend NPU")
    if current_platform.device_count() < dp_size * cp_size:
        pytest.skip(f"Requires {dp_size * cp_size} accelerators")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if current_platform.device_type == "cuda":
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={dp_size * cp_size}",
            "tests/torchrun/run_megatron_loss_aggregation.py",
            f"--cp-size={cp_size}",
            f"--check={check}",
            *(["--prepared"] if prepared else []),
        ],
        env=env,
        cwd=root,
        check=True,
        timeout=240,
    )
