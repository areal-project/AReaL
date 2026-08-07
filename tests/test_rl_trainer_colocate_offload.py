import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType
from areal.trainer.rl_trainer import PPOTrainer


def _config(*, version="v2", mode="awex", use_lora=False, dte=None):
    return SimpleNamespace(
        actor=SimpleNamespace(
            _version=version,
            offload=False,
            weight_update_mode=mode,
            use_lora=use_lora,
            dte=dte,
            scheduling_strategy=SchedulingStrategy(),
        ),
        rollout=SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
            )
        ),
    )


def test_awex_colocation_keeps_rollout_and_actor_offload_enabled():
    trainer = object.__new__(PPOTrainer)
    config = _config()

    offload_rollout, offload_actor = trainer._resolve_actor_rollout_offload(config)

    assert offload_rollout is True
    assert offload_actor is True


def test_dte_colocation_keeps_initial_rollout_weights():
    trainer = object.__new__(PPOTrainer)
    trainer._should_offload_rollout = True
    config = _config(
        dte=SimpleNamespace(
            enabled=True,
            release_initial_rollout_weights=None,
        )
    )

    release_weights = trainer._resolve_release_initial_rollout_weights(config)

    assert release_weights is False


@pytest.mark.parametrize(
    "version,mode,use_lora,expected",
    [
        ("v1", "awex", False, True),
        ("v1", "disk", False, False),
        ("v2", "awex", False, True),
        ("v2", "disk", False, True),
        ("v2", "awex", True, False),
    ],
)
def test_awex_transport_selection(version, mode, use_lora, expected):
    trainer = object.__new__(PPOTrainer)

    assert (
        trainer._uses_awex_weight_update(
            _config(version=version, mode=mode, use_lora=use_lora)
        )
        is expected
    )


@pytest.mark.parametrize(
    "version,use_lora,expected",
    [("v1", False, True), ("v2", False, True), ("v2", True, False)],
)
def test_awex_checkpoint_is_saved_before_weight_update(version, use_lora, expected):
    trainer = object.__new__(PPOTrainer)

    assert (
        trainer._should_save_before_weight_update(
            _config(version=version, use_lora=use_lora)
        )
        is expected
    )


def test_trainer_startup_creates_one_awex_meta_server_and_forwards_it():
    tree = ast.parse(textwrap.dedent(inspect.getsource(PPOTrainer._init_impl)))
    start_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "start_meta_server"
    ]
    awex_meta_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_awex"
    ]

    assert len(start_calls) == 1
    assert awex_meta_calls
    assert all(
        any(keyword.arg == "meta_server_addr" for keyword in call.keywords)
        for call in awex_meta_calls
    )
