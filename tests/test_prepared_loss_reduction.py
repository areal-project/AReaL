# SPDX-License-Identifier: Apache-2.0

"""Full-update unit coefficients survive row-level optimizer schedules."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function
from tests.test_megatron_loss_aggregation import (
    MEAN_MODES,
    independent_loss_oracle,
    make_loss_data,
)

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import pack_tensor_dict, split_training_batch_into_microbatches
from areal.utils.functional.loss_aggregation import PolicyGradientReduction


def _aggregate(reduction, loss, batch):
    return reduction.aggregate(
        loss,
        batch.get("numerator_mask", batch["loss_mask"]),
        denominator_mask=batch["loss_mask"],
        cu_seqlens=batch.get("cu_seqlens"),
        unit_weights=batch.get("unit_weights"),
    )


def _prepare(data, mode):
    return prepare_policy_gradient_batch(
        data,
        mode=mode,
        group_sizes=[3, 1, 1, 3] if mode == "traj-mean" else [4, 4],
    )


@pytest.mark.parametrize("mode", MEAN_MODES)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "scenario", ["ragged", "empty-unit", "filtered", "empty-prompt"]
)
def test_prepared_whole_step_matches_independent_mean(mode, packed, scenario):
    data = make_loss_data(mode, scenario)
    data["attention_mask"] = torch.ones_like(data["loss_mask"])
    step = _prepare(data, mode).for_steps([data])[0]
    parameter = torch.linspace(-0.2, 0.6, 128).reshape(8, 16).requires_grad_()
    loss = (parameter - data["targets"]).square()
    expected = independent_loss_oracle(loss, data, mode)
    batch = pack_tensor_dict(data) if packed else data
    actual = _aggregate(step, loss.flatten() if packed else loss, batch)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter, retain_graph=True)[0],
        torch.autograd.grad(expected, parameter)[0],
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", ["traj-mean", "prompt-mean"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "scenario", ["ragged", "empty-unit", "filtered", "empty-prompt"]
)
@pytest.mark.parametrize(
    "partitions",
    [
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        [[0], [1, 2, 3], [4, 5, 6, 7]],
        [[2, 7], [0, 1, 3, 4, 5, 6]],
    ],
)
def test_unit_mean_across_optimizer_and_forward_splits_matches_update(
    mode, packed, scenario, partitions
):
    data = make_loss_data(mode, scenario)
    data["attention_mask"] = torch.ones_like(data["loss_mask"])
    prepared = _prepare(data, mode)
    batches = [{key: value[rows] for key, value in data.items()} for rows in partitions]
    steps = prepared.for_steps(batches)
    parameter = torch.linspace(-0.2, 0.6, 128).reshape(8, 16).requires_grad_()
    losses = (parameter - data["targets"]).square()
    expected = independent_loss_oracle(losses, data, mode)
    contributions = []
    for rows, batch, step in zip(partitions, batches, steps, strict=True):
        numerator, denominator = 0.0, 0.0
        # A second, forward-level split cuts the same logical units again.
        for offset, row in enumerate(rows):
            fragment = {key: value[offset : offset + 1] for key, value in batch.items()}
            if packed:
                fragment = pack_tensor_dict(fragment)
            reduction = step
            value = losses[row : row + 1]
            local_loss = _aggregate(
                reduction, value.flatten() if packed else value, fragment
            )
            weight = reduction.normalizer_fn(fragment)
            numerator = numerator + local_loss * weight
            denominator = denominator + weight
        torch.testing.assert_close(denominator, torch.tensor(1.0), rtol=1e-6, atol=1e-7)
        contributions.append(numerator / denominator)
    actual = torch.stack(contributions).mean()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter, retain_graph=True)[0],
        torch.autograd.grad(expected, parameter)[0],
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", ["traj-mean", "prompt-mean"])
def test_split_unit_retains_full_update_weight_across_optimizer_steps(mode):
    data = {
        "input_ids": torch.arange(3).reshape(3, 1),
        "attention_mask": torch.ones(3, 1, dtype=torch.bool),
        "loss_mask": torch.ones(3, 1, dtype=torch.bool),
    }
    prepared = prepare_policy_gradient_batch(data, mode=mode, group_sizes=[2, 1])
    batches = split_training_batch_into_microbatches(data, 2)
    steps = prepared.for_steps(batches)
    losses = torch.tensor([[1.0], [1.0], [10.0]], requires_grad=True)
    values = []
    for batch, step in zip(batches, steps, strict=True):
        rows = batch["input_ids"][:, 0]
        values.append(_aggregate(step, losses[rows], batch))
    torch.testing.assert_close(
        torch.stack(values).sort().values,
        torch.tensor([0.5, 10.5]),
        rtol=1e-6,
        atol=1e-6,
    )
    actual = torch.stack(values).mean()
    torch.testing.assert_close(actual, torch.tensor(5.5), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, losses)[0],
        torch.tensor([[0.25], [0.25], [0.5]]),
        rtol=0,
        atol=0,
    )


def test_sequence_mean_keeps_ref_per_step_row_normalization():
    data = {"loss_mask": torch.ones(3, 1, dtype=torch.bool)}
    prepared = prepare_policy_gradient_batch(data, mode="seq-mean")
    batches = [
        {"loss_mask": data["loss_mask"][[0, 2]]},
        {"loss_mask": data["loss_mask"][[1]]},
    ]
    steps = prepared.for_steps(batches)
    losses = [torch.tensor([[1.0], [10.0]]), torch.tensor([[1.0]])]
    values = [
        _aggregate(step, loss, batch)
        for step, batch, loss in zip(steps, batches, losses, strict=True)
    ]
    torch.testing.assert_close(
        torch.stack(values), torch.tensor([5.5, 1.0]), rtol=0, atol=0
    )
    assert all(
        isinstance(step, PolicyGradientReduction) and step.mode == "seq-mean"
        for step, batch in zip(steps, batches, strict=True)
    )


@pytest.mark.parametrize("mode", ["traj-mean", "prompt-mean"])
def test_real_all_masked_step_keeps_positive_weight_and_zero_gradient(mode):
    data = make_loss_data(mode)
    prepared = _prepare(data, mode)
    batches = [
        {key: value[rows] for key, value in data.items()}
        for rows in [[2, 7], [0, 1, 3, 4, 5, 6]]
    ]
    step = prepared.for_steps(batches)[0]
    reduction = step
    values = torch.ones_like(
        batches[0]["loss_mask"], dtype=torch.float32, requires_grad=True
    )
    loss = _aggregate(reduction, values, batches[0])
    torch.testing.assert_close(
        reduction.normalizer_fn(batches[0]), torch.tensor(1.0), rtol=0, atol=0
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(loss, values)[0], torch.zeros_like(values), rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["traj-mean", "prompt-mean"])
def test_prepared_unit_mean_rejects_globally_empty_update(mode):
    data = {"loss_mask": torch.zeros(3, 2, dtype=torch.bool)}
    prepared = prepare_policy_gradient_batch(data, mode=mode, group_sizes=[2, 1])
    with pytest.raises(ValueError, match="active"):
        prepared.for_steps([data])


def test_optimizer_steps_are_capped_by_available_rows():
    data = {"attention_mask": torch.ones(2, 3, dtype=torch.bool)}
    batches = split_training_batch_into_microbatches(data, 8)
    assert len(batches) == 2
    assert all(batch["attention_mask"].shape[0] == 1 for batch in batches)


def test_actor_prompt_mean_uses_actual_group_rows_and_allows_variable_rollouts():
    groups = []
    for trajectories in (8, 7):
        sizes = [1 + i % 2 for i in range(trajectories)]
        n_rows = sum(sizes)
        mask = torch.ones(n_rows, 3, dtype=torch.bool)
        groups.append(
            {
                "input_ids": torch.ones(n_rows, 3, dtype=torch.long),
                "attention_mask": mask,
                "loss_mask": mask,
                "begin_of_trajectory": torch.tensor(
                    [v for size in sizes for v in [1] + [0] * (size - 1)]
                ),
                "rewards": torch.ones(n_rows),
                "advantages": torch.zeros(n_rows, 3),
                "kl_rewards": torch.zeros(n_rows, 3),
                "tot_rewards": torch.zeros(n_rows, 3),
            }
        )
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(
        path="dummy",
        loss_aggregation="prompt-mean",
        ppo_n_minibatches=2,
    )
    actor.m2_threshold = None
    contributions = []

    def train_batch(batch, loss_fn, loss_weight_fn):
        step = loss_fn.keywords["pg_reduction"]
        # Constant per-token loss makes every nonempty prompt's mean exactly one.
        contributions.append(
            _aggregate(
                step, torch.ones_like(batch["loss_mask"], dtype=torch.float32), batch
            )
        )
        return {}

    actor.engine = SimpleNamespace(
        train=lambda: None,
        get_version=lambda: 0,
        data_parallel_group=None,
        device="cpu",
        train_batch=train_batch,
    )
    with patch("areal.trainer.ppo.actor.stats_tracker"):
        actor.ppo_update(groups)
    assert len(contributions) == 2
    torch.testing.assert_close(
        torch.stack(contributions).mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("backend", ["fsdp", "archon"])
def test_prepared_weights_are_loss_metadata_not_model_kwargs(backend):
    data = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "unit_weights": torch.full((1, 4), 0.25),
    }
    item = SimpleNamespace(orig_mb=dict(data), padded_mb=dict(data), padding_length=0)
    namespace = {
        "torch": torch,
        "FSDPTrainContext": SimpleNamespace,
        "ArchonTrainContext": SimpleNamespace,
    }
    path = (
        "areal/engine/fsdp_engine.py"
        if backend == "fsdp"
        else "areal/experimental/engine/archon_engine.py"
    )
    prepare = _load_function(path, "_prepare_mb_inputs", namespace)
    engine = SimpleNamespace(
        parallel_helper=SimpleNamespace(sp_size=1),
        parallel_dims=SimpleNamespace(cp_enabled=False),
        enable_tree_training=False,
    )
    inputs, context = prepare(engine, item)
    assert "unit_weights" not in inputs
    assert "unit_weights" in item.padded_mb
    torch.testing.assert_close(
        context.mb_input["unit_weights"], data["unit_weights"], rtol=0, atol=0
    )


@pytest.mark.parametrize("use_sapo", [False, True])
def test_actor_loss_consumes_prepared_mode_free_reduction(use_sapo):
    mask = torch.ones(2, 2, dtype=torch.bool)
    logprobs = torch.zeros(2, 2, requires_grad=True)
    data = {
        "logprobs": torch.zeros_like(logprobs),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": mask,
        "prox_logp": torch.zeros_like(logprobs),
    }

    class DoubleTokenMean:
        def __bool__(self):
            return False

        def normalizer_fn(self, data):
            return PolicyGradientReduction().normalizer_fn(data)

        def aggregate(self, loss, numerator_mask, **kwargs):
            return 2 * PolicyGradientReduction().aggregate(
                loss, numerator_mask, **kwargs
            )

    with patch("areal.trainer.ppo.actor.stats_tracker"):
        loss = grpo_loss_fn(
            logprobs,
            torch.zeros_like(logprobs),
            data,
            0.2,
            None,
            None,
            use_sapo_loss=use_sapo,
            pg_reduction=DoubleTokenMean(),
        )
    torch.testing.assert_close(
        loss, torch.tensor(-4.0 if use_sapo else -2.0), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        torch.autograd.grad(loss, logprobs)[0],
        -2 * mask.float() / 4,
        rtol=1e-6,
        atol=1e-6,
    )
