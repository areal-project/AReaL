# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.api.cli_args import SaverConfig
from areal.api.io_struct import FinetuneSpec
from areal.utils.saver import Saver


def _make_saver(tmp_path) -> Saver:
    config = SaverConfig(
        experiment_name="test_exp",
        trial_name="test_trial",
        fileroot=str(tmp_path),
        freq_steps=1,
        mode="sync",
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=1,
        train_batch_size=1,
    )
    saver = Saver(config, ft_spec)
    saver._should_use_async = Mock(return_value=False)
    return saver


def test_save_without_base_model_path_uses_engine_config_path(tmp_path):
    """Periodic HF saves retain source assets when callers omit the base path."""
    saver = _make_saver(tmp_path)
    source_path = tmp_path / "source-checkpoint"
    source_path.mkdir()
    engine = Mock()
    engine.config = SimpleNamespace(path=str(source_path))

    saver.save(engine, epoch=0, step=0, global_step=0)

    meta = engine.save.call_args.args[0]
    assert meta.base_model_path == str(source_path)


def test_save_with_base_model_path_preserves_explicit_override(tmp_path):
    """An explicit source path takes precedence over the engine configuration."""
    saver = _make_saver(tmp_path)
    engine = Mock()
    engine.config = SimpleNamespace(path="/models/source-checkpoint")

    saver.save(
        engine,
        epoch=0,
        step=0,
        global_step=0,
        base_model_path="/models/explicit-checkpoint",
    )

    meta = engine.save.call_args.args[0]
    assert meta.base_model_path == "/models/explicit-checkpoint"


def test_save_with_hub_model_path_does_not_treat_it_as_local_directory(tmp_path):
    """Hub model IDs do not enter the local Hugging Face asset-copy path."""
    saver = _make_saver(tmp_path)
    engine = Mock()
    engine.config = SimpleNamespace(path="organization/model")

    saver.save(engine, epoch=0, step=0, global_step=0)

    meta = engine.save.call_args.args[0]
    assert meta.base_model_path is None


def test_save_with_source_matching_destination_preserves_source_path(tmp_path):
    """The exporter receives the source path so it can snapshot config fields."""
    saver = _make_saver(tmp_path)
    save_path = Saver.get_model_save_path(
        "test_exp",
        "test_trial",
        str(tmp_path),
        epoch=0,
        step=0,
        globalstep=0,
    )
    engine = Mock()
    engine.config = SimpleNamespace(path=save_path)

    saver.save(engine, epoch=0, step=0, global_step=0)

    meta = engine.save.call_args.args[0]
    assert meta.base_model_path == str(save_path)


def _saved_global_steps(engine: Mock) -> list[int]:
    return [
        int(call.args[0].path.rsplit("globalstep", 1)[1])
        for call in engine.save.call_args_list
    ]


@pytest.mark.parametrize(
    ("freq_kwargs", "expected_steps"),
    [
        ({"freq_steps": 2}, [1, 3, 5]),
        ({"freq_epochs": 2}, [5]),
    ],
)
def test_save_applies_frequency_once_per_step_for_all_models(
    tmp_path, freq_kwargs, expected_steps
):
    """Actor and critic saved in the same step share one frequency decision."""
    config = SaverConfig(
        experiment_name="test_exp",
        trial_name="test_trial",
        fileroot=str(tmp_path),
        mode="sync",
        **freq_kwargs,
    )
    ft_spec = FinetuneSpec(total_train_epochs=2, dataset_size=3, train_batch_size=1)
    saver = Saver(config, ft_spec)
    saver._should_use_async = Mock(return_value=False)
    actor, critic = Mock(), Mock()
    actor.config = critic.config = SimpleNamespace(path=None)

    global_step = 0
    for epoch in range(ft_spec.total_train_epochs):
        for step in range(ft_spec.steps_per_epoch):
            saver.save(actor, epoch, step, global_step)
            saver.save(critic, epoch, step, global_step, name="critic")
            global_step += 1

    assert _saved_global_steps(actor) == expected_steps
    assert _saved_global_steps(critic) == expected_steps
