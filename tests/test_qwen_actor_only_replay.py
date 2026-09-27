# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from examples.swe.qwen38_flash_next.train_rl import run_actor_only_replay


def test_actor_only_replay_updates_saved_batch_without_rollout(
    tmp_path, monkeypatch
) -> None:
    from areal.api.alloc_mode import ModelAllocation
    from areal.engine import MegatronPPOActor
    from areal.infra.scheduler.slurm import SlurmScheduler

    snapshot = tmp_path / "compute_advantages-0000.output.pt"
    batch = [{"rewards": torch.tensor([1.0])}]
    torch.save(
        {
            "schema_version": 2,
            "metadata": {
                "method": "compute_advantages",
                "model_path": "model",
                "allocation_mode": "allocation",
                "n_samples": 12,
            },
            "batch": batch,
        },
        snapshot,
    )
    config = SimpleNamespace(
        total_train_steps=1,
        recover=SimpleNamespace(mode="off"),
        evaluator=SimpleNamespace(eval_before_train=False),
        tokenizer_path="model",
        allocation_mode="allocation",
        gconfig=SimpleNamespace(n_samples=12),
        actor=SimpleNamespace(backend="backend"),
        train_dataset=SimpleNamespace(batch_size=8),
    )
    actor = MagicMock()
    scheduler = MagicMock()
    parallel = object()
    monkeypatch.setenv("QWEN_MOUNTS", "/storage:/storage")
    monkeypatch.setattr(SlurmScheduler, "__new__", lambda cls, **kwargs: scheduler)
    monkeypatch.setattr(
        MegatronPPOActor, "as_controller", staticmethod(lambda *_args: actor)
    )
    monkeypatch.setattr(
        ModelAllocation,
        "from_str",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(parallel=parallel)),
    )

    run_actor_only_replay(config, snapshot)

    actor.create_process_group.assert_called_once_with(parallel_strategy=parallel)
    actor.ppo_update.assert_called_once()
    torch.testing.assert_close(
        actor.ppo_update.call_args.args[0][0]["rewards"],
        batch[0]["rewards"],
        rtol=0,
        atol=0,
    )
    actor.step_lr_scheduler.assert_called_once_with()
    actor.destroy.assert_called_once_with()
