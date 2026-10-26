# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from examples.profile.calibration_cases import (
    ACTOR_CASES,
    ROLLOUT_CASES,
    ActorCase,
    make_actor_batch,
    select_cases,
)

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor


def test_actor_observer_preserves_nested_execution_and_version(tmp_path, monkeypatch):
    """Per-rank observations identify nested work without changing engine outputs."""
    import json

    from areal.elastic.monitor.actor import install_actor_observer

    class Engine:
        def get_version(self):
            return 7

        def _prepare_mb_list(self):
            return SimpleNamespace(
                group_lens=[8, 16], padded_to_lengths=[8, 16], padding_lengths=[0, 0]
            )

        def optimizer_step(self):
            return {"update_successful": 1.0, "grad_norm": 2.0}

        def ppo_update(self):
            self._prepare_mb_list()
            return self.optimizer_step()

    monkeypatch.setenv("AREAL_ROLLOUT_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    engine = Engine()
    install_actor_observer(engine)
    assert engine.ppo_update() == {"update_successful": 1.0, "grad_norm": 2.0}
    rows = [
        json.loads(line)
        for p in tmp_path.glob("actor-phases-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(rows) == 3
    root = next(row for row in rows if row["operation"] == "ppo_update")
    assert all(
        row["weights_version"] == 7 and row["root_phase"] == "ppo_update"
        for row in rows
    )
    assert all(
        row["parent_call_id"] == root["call_id"] for row in rows if row is not root
    )
    assert rows[0]["microbatches"]["real_tokens"] == [8, 16]


def test_actor_catalog_has_equal_token_different_attention_control():
    cases = {case.name: case for case in ACTOR_CASES}
    mixed, uniform = cases["mixed"].lengths, cases["batch-4"].lengths
    assert len(mixed) == len(uniform) == 4
    assert sum(mixed) == sum(uniform) == 16384
    assert sum(n * n for n in mixed) > sum(n * n for n in uniform)
    assert len(ROLLOUT_CASES) == 16
    assert [case.name for case in select_cases("actor", "batch-4,length-4096")] == [
        "batch-4",
        "length-4096",
    ]


def test_actor_inputs_are_reproducible_and_keep_nonzero_ppo_signal():
    case = ActorCase("small", (16, 32))
    batch = make_actor_batch(case, seed=7, dp_rank=0)
    repeated = make_actor_batch(case, seed=7, dp_rank=0)
    config = PPOActorConfig(
        ppo_n_minibatches=1,
        recompute_logprob=True,
        use_decoupled_loss=True,
        reward_bias=-0.5,
        reward_scaling=10,
        kl_ctl=0,
        mask_no_eos_with_zero=False,
    )
    actor = PPOActor(config, SimpleNamespace())
    for item, copy in zip(batch, repeated):
        torch.testing.assert_close(item["input_ids"], copy["input_ids"], rtol=0, atol=0)
        length = item["input_ids"].shape[-1]
        assert item["loss_mask"].sum() == length // 8
        item["prox_logp"] = -torch.arange(length, dtype=torch.float32).reshape(1, -1)
        item["logprobs"] = torch.roll(item["prox_logp"], 1, -1) * item["loss_mask"]
    result = actor.compute_advantages(batch)
    for item in result:
        mask = item["loss_mask"].bool()
        assert torch.isfinite(item["advantages"]).all()
        assert item["advantages"][mask].abs().sum() > 0
        torch.testing.assert_close(
            item["logprobs"][mask],
            item["prox_logp"][mask],
            rtol=0,
            atol=0,
        )
