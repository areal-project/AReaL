# SPDX-License-Identifier: Apache-2.0
"""Check required recovery at the recipe's actual training boundary."""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest

from examples.swe.qwen38_flash_next import train_rl


@dataclass
class _ArenaConfig:
    arena_task_envs: dict[str, str] = field(default_factory=dict)
    timeout: int = 60


@dataclass
class _DatasetConfig:
    batch_size: int = 1
    shuffle: bool = True
    drop_last: bool = True


@pytest.mark.parametrize(
    ("profile", "require_recover", "checkpoint_loaded", "should_fail"),
    [
        ("swe", None, False, False),
        ("swe", "0", False, False),
        ("swe", "1", False, True),
        ("swe", "1", True, False),
        ("swe-eval", "1", False, False),
    ],
)
def test_recipe_required_recovery_checks_loaded_state_before_training(
    monkeypatch, tmp_path, profile, require_recover, checkpoint_loaded, should_fail
):
    events = []
    recovered_state = (
        SimpleNamespace(last_step_info=SimpleNamespace(global_step=5))
        if checkpoint_loaded
        else None
    )
    config = SimpleNamespace(
        experiment_name="arena-recovery-test",
        trial_name="trial",
        econfig=_ArenaConfig(),
        train_dataset=_DatasetConfig(),
        gconfig=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=1000000, max_new_tokens=42, n_samples=12
        ),
        eval_gconfig=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=1000000, max_new_tokens=42, n_samples=12
        ),
        recover=SimpleNamespace(mode="auto"),
        evaluator=SimpleNamespace(eval_before_train=False),
        rollout=SimpleNamespace(),
        actor=SimpleNamespace(
            optimizer=SimpleNamespace(lr=1e-5),
            mb_spec=SimpleNamespace(n_mbs=1, n_mbs_divisor=1),
        ),
        should_accept_fn="filter",
    )

    class Trainer:
        def __init__(self, config, **kwargs):
            self.recover_info = recovered_state
            events.append("initialized")

        def __enter__(self):
            events.append("entered")
            return self

        def __exit__(self, *args):
            events.append("closed")
            return False

        def train(self, **kwargs):
            events.append("trained")

    class RemoteEngine:
        @staticmethod
        def as_controller(*args, **kwargs):
            return object()

    original_factory = RemoteEngine.as_controller
    imports = {
        "examples.swe.train_swe_rl": {
            "get_arena_mixture_dataset": lambda *args, **kwargs: (
                [{"data_id": "env:task@v1"}],
                ["stream"],
            )
        },
        "examples.swe.utils": {"SWEPPOConfig": object},
        "areal": {"PPOTrainer": Trainer},
        "areal.api.cli_args": {
            "load_expr_config": lambda *args: (config, None),
            "ValidDatasetConfig": _DatasetConfig,
        },
        "areal.engine.sglang_remote": {"RemoteSGLangEngine": RemoteEngine},
        "areal.infra.scheduler.slurm": {"SlurmScheduler": object},
        "datasets": {"Dataset": SimpleNamespace(from_list=lambda rows: rows)},
    }
    for name, attributes in imports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("QWEN_ARENA_TASK_IDS_FILE", raising=False)
    if profile == "swe-eval":
        selection = tmp_path / "tasks.json"
        selection.write_text('["env:task@v1"]')
        monkeypatch.setenv("QWEN_ARENA_TASK_IDS_FILE", str(selection))
    if require_recover is None:
        monkeypatch.delenv("QWEN_REQUIRE_RECOVER", raising=False)
    else:
        monkeypatch.setenv("QWEN_REQUIRE_RECOVER", require_recover)

    if should_fail:
        with pytest.raises(RuntimeError, match="no checkpoint was loaded"):
            train_rl.main(profile, [])
        assert events == ["initialized", "entered", "closed"]
    else:
        train_rl.main(profile, [])
        assert events == ["initialized", "entered", "trained", "closed"]
    assert RemoteEngine.as_controller is original_factory
    if profile == "swe-eval":
        assert config.recover.mode == "disabled"
        assert config.total_train_steps == 0
