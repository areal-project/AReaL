# SPDX-License-Identifier: Apache-2.0

"""Exercise the migrated example against main's real dataset routing API."""

import copy
import json
from contextlib import ExitStack

import pytest
from omegaconf import OmegaConf

from examples.prefix_replay.config import PrefixReplayOPDConfig
from examples.prefix_replay.train_opd import load_replay_dataset

from areal.api.cli_args import to_structured_cfg
from areal.dataset.mopd import ROUTE_METADATA_KEY, DatasetRoute


@pytest.fixture
def replay_config(monkeypatch, tmp_path):
    """Resolve the actual recipe without accessing model files or a GPU."""
    for key, value in {
        "MOPD_STUDENT_MODEL_PATH": "/models/student",
        "MOPD_TEACHER_MODEL_PATH": "/models/teacher",
        "PREFIX_REPLAY_DATA_PATH": str(tmp_path / "replay.jsonl"),
        "AREAL_ADMIN_API_KEY": "test-only-non-default-key",
        "AREAL_IMAGE": "areal:test",
        "AREAL_FILEROOT": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    raw = OmegaConf.load("examples/prefix_replay/local.yaml")
    config = OmegaConf.to_object(to_structured_cfg(raw, PrefixReplayOPDConfig))
    assert config.mopd.teacher_groups == {"replay": {"qwen3_14b": 1.0}}
    assert config.gconfig.drop_incomplete_group
    assert config.mopd.loss.rl_coefficient == 0
    config.prefix_replay.kappa = 1.0
    path = config.train_dataset.sources[0].path
    with open(path, "w") as output:
        for route in ("game", "code"):
            output.write(
                json.dumps(
                    {
                        "instance_id": route,
                        "task_type": route,
                        "messages": [
                            {"role": "user", "content": route},
                            {"role": "assistant", "content": "action"},
                        ],
                    }
                )
                + "\n"
            )
    # Test preprocessing and source routing using a deterministic template.
    monkeypatch.setattr(
        "areal.utils.hf_utils.apply_chat_template",
        lambda *args, **kwargs: [1, 2, 3] if kwargs.get("tokenize") else "prompt",
    )
    return config


def test_replay_source_cache_cold_hot_and_disabled_match(replay_config, monkeypatch):
    """Cache hits retain main's typed route and skip preprocessing."""

    def load():
        with ExitStack() as stack:
            data = load_replay_dataset(
                replay_config, object(), split="train", stack=stack, cache_dirs=set()
            )
            return [data[i] for i in range(len(data))]

    cold = load()
    assert len(cold) == 2
    assert cold[0][ROUTE_METADATA_KEY] == DatasetRoute(0, "replay")
    with monkeypatch.context() as patch:
        patch.setattr(
            "examples.prefix_replay.train_opd.load_prefix_replay_indexed_dataset",
            lambda *a, **kw: pytest.fail("hot cache rebuilt"),
        )
        assert load() == cold
    monkeypatch.setenv("PREFIX_REPLAY_DISABLE_CACHE", "1")
    assert load() == cold


def test_replay_mixed_file_route_selection_preserves_source_provenance(replay_config):
    """Route filtering happens before main attaches its source-owned route."""
    source = replay_config.train_dataset.sources[0]
    source.dataset_kwargs = {"route_field": "task_type", "route_value": "code"}
    second = copy.deepcopy(source)
    second.dataset_kwargs["route_value"] = "game"
    replay_config.train_dataset.sources.append(second)
    with ExitStack() as stack:
        data = load_replay_dataset(
            replay_config, object(), split="train", stack=stack, cache_dirs=set()
        )
        assert len(data) == 2
        assert data[0]["instance_id"] == "code"
        assert data[0][ROUTE_METADATA_KEY] == DatasetRoute(0, "replay")


def test_replay_unknown_selected_route_fails_before_training(replay_config):
    """An empty selected source cannot silently disappear from training."""
    replay_config.train_dataset.sources[0].dataset_kwargs = {
        "route_field": "task_type",
        "route_value": "missing",
    }
    with ExitStack() as stack, pytest.raises(ValueError, match="no prefixes"):
        load_replay_dataset(
            replay_config, object(), split="train", stack=stack, cache_dirs=set()
        )


def test_replay_main_wires_pure_distillation_and_total_budget(
    replay_config, monkeypatch
):
    """Run the entry point through dataset setup and inspect its trainer contract."""
    from examples.prefix_replay import train_opd

    import areal

    calls = {}

    class RecordingTrainer:
        def __init__(self, config, **datasets):
            calls.update(datasets)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def train(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(areal, "PPOTrainer", RecordingTrainer)
    monkeypatch.setattr(train_opd, "load_expr_config", lambda *a: (replay_config, ""))
    monkeypatch.setattr(train_opd, "load_hf_tokenizer", lambda *a: object())
    replay_config.rollout.agent.engine_max_tokens = 4096
    train_opd.main([])
    assert calls["valid_dataset"] is None
    assert len(calls["train_dataset"]) == 2
    assert calls["workflow_kwargs"]["n"] == 1
    assert calls["workflow_kwargs"]["extra_body"]["max_total_tokens"] == 4096
    assert calls["dynamic_filter_fn"] is None


def test_replay_unconfigured_mixed_route_fails(replay_config):
    """Do not silently lose a route when migrating a mixed replay file."""
    replay_config.train_dataset.sources[0].dataset_kwargs = {
        "route_field": "task_type",
        "route_value": "code",
    }
    with (
        ExitStack() as stack,
        pytest.raises(ValueError, match="Unconfigured replay routes"),
    ):
        load_replay_dataset(
            replay_config, object(), split="train", stack=stack, cache_dirs=set()
        )
