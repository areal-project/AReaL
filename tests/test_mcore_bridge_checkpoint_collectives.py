# SPDX-License-Identifier: Apache-2.0

"""CPU save contracts: real Gloo error propagation and export config isolation."""

import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from transformers import GenerationConfig, PretrainedConfig

from areal.models.mcore.mcore_bridge_adapter import MCoreBridgeAdapter
from areal.models.mcore.mcore_bridge_checkpoint import (
    _preserve_hf_auxiliary_files,
    finalize_mcore_bridge_checkpoint,
)


class _SourceConfig(PretrainedConfig):
    model_type = "qwen4_exp"


def _source_config():
    return _SourceConfig(
        text_config=PretrainedConfig(
            mtp={
                "hybrid": True,
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
            },
            mtp_num_hidden_layers=1,
            mtp_use_dedicated_embeddings=False,
        )
    )


def test_auxiliary_assets_fill_tokenizer_files_without_overwriting_saved_config(
    tmp_path,
):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "tokenizer.json").write_text('{"version":"1.0"}')
    (source / "tokenizer_config.json").write_text('{"source":true}')
    (output / "tokenizer_config.json").write_text('{"exported":true}')

    report = _preserve_hf_auxiliary_files(str(source), str(output))

    assert report["copied"] == ["tokenizer.json"]
    assert report["preserved_existing"] == ["tokenizer_config.json"]
    assert (output / "tokenizer.json").read_bytes() == (
        source / "tokenizer.json"
    ).read_bytes()
    assert json.loads((output / "tokenizer_config.json").read_text()) == {
        "exported": True
    }


def test_auxiliary_assets_preserve_generation_config_and_template(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    GenerationConfig(eos_token_id=2, pad_token_id=0, max_new_tokens=16).save_pretrained(
        source
    )
    (source / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    (source / "preprocessor_config.json").write_text('{"patch_size": 16}')
    (source / "video_preprocessor_config.json").write_text('{"temporal_patch_size": 2}')
    (source / "config.json").write_text('{"mtp_num_hidden_layers": 1}')
    (source / "unrelated.json").write_text("{}")
    (output / "config.json").write_text('{"mtp_num_hidden_layers": 0}')

    report = _preserve_hf_auxiliary_files(str(source), str(output))

    reloaded = GenerationConfig.from_pretrained(output)
    assert reloaded.eos_token_id == 2
    assert reloaded.max_new_tokens == 16
    for name in report["copied"]:
        assert (source / name).read_bytes() == (output / name).read_bytes()
    assert set(report["copied"]) == {
        "generation_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    }
    assert (
        json.loads((output / "config.json").read_text())["mtp_num_hidden_layers"] == 0
    )
    assert not (output / "unrelated.json").exists()


def test_auxiliary_assets_never_override_exported_template_or_processor(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("chat_template.jinja", "preprocessor_config.json"):
        (source / name).write_text("original")
        (output / name).write_text("exported")

    report = _preserve_hf_auxiliary_files(str(source), str(output))

    assert report == {
        "copied": [],
        "preserved_existing": ["chat_template.jinja", "preprocessor_config.json"],
    }
    assert (output / "chat_template.jinja").read_text() == "exported"
    assert (output / "preprocessor_config.json").read_text() == "exported"


@pytest.mark.parametrize("dangling", [False, True])
def test_auxiliary_assets_reject_output_symlink_without_touching_target(
    tmp_path, dangling
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "chat_template.jinja").write_text("source template")
    outside = tmp_path / "outside.jinja"
    if not dangling:
        outside.write_text("outside template")
    (output / "chat_template.jinja").symlink_to(outside)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _preserve_hf_auxiliary_files(str(source), str(output))

    assert (output / "chat_template.jinja").is_symlink()
    if dangling:
        assert not outside.exists()
    else:
        assert outside.read_text() == "outside template"


def test_auxiliary_assets_reject_directory_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "chat_template.jinja").write_text("template")
    target = output / "chat_template.jinja"
    target.mkdir()

    with pytest.raises(ValueError, match="must be a regular file"):
        _preserve_hf_auxiliary_files(str(source), str(output))

    assert list(target.iterdir()) == []


class _TokenizerWriter:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def save_pretrained(self, path):
        if self.fail:
            raise OSError("tokenizer save failed")
        (Path(path) / "tokenizer_config.json").write_text("{}")


class _TensorSaveAlreadyFinished:
    hf_layers_prefix = "model.language_model.layers"

    def save_weights(self, models, path):
        # The real bridge has returned from its own tensor-save collectives.
        # This double exercises only the following CPU validation contract.
        pass


def _checkpoint_collective_worker(rank: int, directory: str):
    root = Path(directory)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = dist.group.WORLD
    try:
        config = _source_config()
        before = config.to_dict()
        for name, expected_error in (
            ("success", None),
            ("missing_text", "missing required non-restorable"),
            ("tokenizer_error", "tokenizer save failed"),
        ):
            try:
                report = finalize_mcore_bridge_checkpoint(
                    str(root / "source"),
                    str(root / name),
                    hf_config=config,
                    language_model_only=True,
                    mtp_enabled=False,
                    cpu_group=group,
                    tokenizer=_TokenizerWriter(fail=name == "tokenizer_error"),
                )
            except RuntimeError as exc:
                assert expected_error is not None and expected_error in str(exc)
            else:
                assert expected_error is None
                assert report["restored_keys"] == ["model.visual.patch_embed.weight"]
                assert report["omitted_mtp_keys"] == ["mtp.fc_hidden.weight"]
                saved = json.loads((root / name / "config.json").read_text())
                assert saved["text_config"]["mtp"] is None
                assert saved["text_config"]["mtp_num_hidden_layers"] == 0
                assert (root / name / "tokenizer_config.json").is_file()
                assert (
                    root / name / "chat_template.jinja"
                ).read_text() == "{{ messages[0]['content'] }}"
                assert (
                    GenerationConfig.from_pretrained(root / name).max_new_tokens == 16
                )
            assert config.to_dict() == before
            # Both success and error paths leave peers able to do another collective.
            dist.barrier(group=group)

        adapter = MCoreBridgeAdapter.__new__(MCoreBridgeAdapter)
        adapter.config = SimpleNamespace(
            hf_model_type="qwen4_exp",
            ngram_size=3,
            heads_per_ngram=1,
            ple_embed_dim=4,
            split_ngram_parts=1,
            make_ngram_vocab_size_divisible_by=4,
            ple_layer_ids=[] if rank == 0 else [1],
        )
        adapter.bridge = _TensorSaveAlreadyFinished()
        try:
            adapter.save_weights([], str(root / "success"), cpu_group=group)
        except RuntimeError as exc:
            assert "rank 1" in str(exc)
            assert "Missing required PLE checkpoint tensor" in str(exc)
        else:
            raise AssertionError("A single-rank PLE failure must reach every rank")
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_checkpoint_finalization_errors_reach_all_cpu_ranks(tmp_path):
    pytest.importorskip("mcore_bridge.utils.qwen4_exp_checkpoint")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    GenerationConfig(eos_token_id=2, max_new_tokens=16).save_pretrained(source)
    (source / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    save_file(
        {
            "lm_head.weight": torch.ones(2),
            "model.visual.patch_embed.weight": torch.ones(3),
            "mtp.fc_hidden.weight": torch.ones(2),
        },
        source / "model.safetensors",
    )
    for name in ("success", "missing_text", "tokenizer_error"):
        output = tmp_path / name
        output.mkdir()
        weights = (
            {} if name == "missing_text" else {"lm_head.weight": torch.full((2,), 7.0)}
        )
        save_file(weights, output / "model.safetensors")
    context = mp.spawn(
        _checkpoint_collective_worker, args=(str(tmp_path),), nprocs=2, join=False
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU checkpoint collective timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
