# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

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
