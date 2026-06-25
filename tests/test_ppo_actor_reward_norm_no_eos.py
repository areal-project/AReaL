# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import types
from types import SimpleNamespace

import torch


def _install_import_shims() -> None:
    try:
        import torch.distributed.checkpoint.staging as staging

        staging.DefaultStager = getattr(
            staging, "DefaultStager", type("DefaultStager", (), {})
        )
        staging.StagingOptions = getattr(
            staging, "StagingOptions", type("StagingOptions", (), {})
        )
    except Exception:
        pass

    try:
        import torch.distributed.checkpoint.state_dict_saver as saver

        saver.AsyncSaveResponse = getattr(
            saver, "AsyncSaveResponse", type("AsyncSaveResponse", (), {})
        )
    except Exception:
        pass

    for name in ("swanlab", "trackio", "tabulate"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            if name == "tabulate":
                module.tabulate = lambda *args, **kwargs: ""
            sys.modules[name] = module

    if "tensorboardX" not in sys.modules:
        module = types.ModuleType("tensorboardX")
        module.__spec__ = importlib.machinery.ModuleSpec("tensorboardX", None)
        module.SummaryWriter = type(
            "SummaryWriter", (), {"__init__": lambda self, *a, **k: None}
        )
        sys.modules["tensorboardX"] = module


_install_import_shims()

from areal.api.cli_args import NormConfig  # noqa: E402
from areal.trainer.ppo.actor import PPOActor  # noqa: E402
from areal.utils.data import KLEstimator, Normalization  # noqa: E402


def _make_actor() -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.config = SimpleNamespace(
        overlong_reward_penalty=False,
        use_decoupled_loss=False,
        recompute_logprob=False,
        mask_no_eos_with_zero=True,
    )
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 1000.0
    actor.reward_norm = Normalization(NormConfig(mean_level="batch", std_level=None))
    actor.adv_norm = None
    actor.kl_ctl = 0.0
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.mask_no_eos_with_zero = True
    return actor


def _run_case(actor: PPOActor, rewards: list[float]) -> dict[str, torch.Tensor]:
    data = {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.bool),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "logprobs": torch.zeros(2, 3),
    }
    return actor._compute_advantages(
        {key: value.clone() for key, value in data.items()}
    )


def test_reward_norm_excludes_no_eos_rows_masked_from_task_reward() -> None:
    actor = _make_actor()

    baseline = _run_case(actor, [1.0, 1.0])
    with_no_eos_outlier = _run_case(actor, [1.0, 100.0])

    torch.testing.assert_close(
        with_no_eos_outlier["tot_rewards"][0], baseline["tot_rewards"][0]
    )
    torch.testing.assert_close(
        with_no_eos_outlier["advantages"][0], baseline["advantages"][0]
    )
    torch.testing.assert_close(
        with_no_eos_outlier["tot_rewards"][1],
        torch.zeros_like(with_no_eos_outlier["tot_rewards"][1]),
    )
