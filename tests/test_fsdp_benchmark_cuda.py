# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys

import pytest
import torch

from areal.tools.benchmark_fsdp import run_process


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [1, 2])
def test_fsdp_sweep_runs_real_optimizer_steps(tmp_path, monkeypatch, world_size):
    """Exercise packed FA2/FSDP with local random weights and no downloads."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("Requires FlashAttention 2")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config

    model = tmp_path / "model"
    model.mkdir()
    model_config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    model_config.save_pretrained(model)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
                **{f"t{i}": i for i in range(4, 128)},
            },
            unk_token="[UNK]",
        )
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(model)
    output = tmp_path / "results"
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    cli_log = tmp_path / "cli.log"
    completed = run_process(
        [
            sys.executable,
            "-m",
            "areal.tools.benchmark_fsdp",
            "--model-path",
            str(model),
            "--init-from-scratch",
            "--seq-lens",
            "16,16,16,16",
            "--token-budgets",
            "16,32" if world_size == 1 else "16",
            "--gradient-checkpointing",
            "both" if world_size == 1 else "off",
            "--nproc-per-node",
            str(world_size),
            "--warmup-steps",
            "1",
            "--steps",
            "1" if world_size == 2 else "2",
            "--timeout",
            "180",
            "--output-dir",
            str(output),
        ],
        cli_log,
        timeout=780,
    )
    if completed["returncode"] or completed["status"] != "completed":
        logs = "\n".join(p.read_text() for p in output.glob("*/output.log"))
        pytest.fail(cli_log.read_text() + logs)
    report = json.loads((output / "report.json").read_text())
    assert len(report["trials"]) == (4 if world_size == 1 else 1)
    assert report["plan"]["input_tokens_per_step"] == 64
    assert report["plan"]["loss_tokens_per_step"] == 60
    digests = set()
    for trial in report["trials"]:
        assert trial["status"] == "ok"
        assert trial["input_tokens_per_second"] > 0
        assert len(trial["ranks"]) == world_size
        for rank in trial["ranks"]:
            digests.add(rank["global_batch_sha256"])
            assert all(s["update_successful"] == 1 for s in rank["samples"])
    assert len(digests) == 1
