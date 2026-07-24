from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.profile.postprocess_profile import postprocess_profile
from examples.profile.profile_rank_utils import (
    pp_rank0_ranks_from_backend,
    resolve_profile_ranks,
)
from examples.profile.train_sft_profile import RepeatedProfileSFTDataset

from areal.utils.functional.vocab_parallel import _resolve_chunk_size


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [(ord(ch) % 97) + 3 for ch in text if not ch.isspace()]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for event in events:
            json.dump(event, fout)
            fout.write("\n")


def test_repeated_profile_sft_dataset_builds_fixed_loss_span() -> None:
    dataset = RepeatedProfileSFTDataset(
        _FakeTokenizer(),
        seq_len=32,
        dataset_size=3,
        loss_start_ratio=0.5,
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["input_ids"].shape == sample["loss_mask"].shape == (32,)
    assert sample["loss_mask"][:16].sum().item() == 0
    assert sample["loss_mask"][16:].sum().item() == 16
    assert dataset[0]["input_ids"].data_ptr() != dataset[1]["input_ids"].data_ptr()


def test_logprobs_chunk_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AREAL_LOGPROBS_CHUNK_SIZE", raising=False)
    assert _resolve_chunk_size(1024) == 1024

    monkeypatch.setenv("AREAL_LOGPROBS_CHUNK_SIZE", "128")
    assert _resolve_chunk_size(1024) == 128

    monkeypatch.setenv("AREAL_LOGPROBS_CHUNK_SIZE", "0")
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_chunk_size(1024)


def test_profile_rank_utils_resolves_pp_rank0(tmp_path: Path) -> None:
    config = tmp_path / "profile.yaml"
    config.write_text(
        'actor:\n  backend: "megatron:(attn:d1p2t2c2|ffn:d1p2e4)"\n',
        encoding="utf-8",
    )

    assert pp_rank0_ranks_from_backend("megatron:d2p1t2") == "0"
    assert pp_rank0_ranks_from_backend("megatron:(attn:d1p2t2c2|ffn:d1p2e4)") == "0,4"
    assert resolve_profile_ranks("pp_rank0", config) == "0,4"
    assert resolve_profile_ranks("", config) == ""
    assert resolve_profile_ranks("0,2-4", config) == "0,2-4"


def test_postprocess_profile_writes_kernel_views_and_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    trace_dir = log_dir / "perf_tracer" / "actor"
    snapshot_dir = log_dir / "memory_snapshots" / "step_1"
    run_dir = tmp_path / "run"
    trace_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "snapshot_rank00_p0d0c0t0.pickle").write_bytes(b"snapshot")
    trainer_log = log_dir / "trainer.log"
    trainer_log.write_text(
        "memory allocated (GB): 1.25/2.0, memory reserved (GB): 2.50/3.0\n"
        "device memory used/total (GB): 3.75/80.0\n",
        encoding="utf-8",
    )
    nvidia_smi = run_dir / "nvidia_smi.csv"
    run_dir.mkdir()
    nvidia_smi.write_text(
        "timestamp,index,memory.used [MiB]\nnow,0,123\nnow,0,456\n",
        encoding="utf-8",
    )
    _write_jsonl(
        trace_dir / "traces-r0.jsonl",
        [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "dur": 1,
                "args": {"rank": 0, "correlation": 7},
            },
            {
                "ph": "s",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 7,
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "args": {"rank": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "profile_kernel",
                "pid": 1,
                "tid": -3,
                "ts": 2,
                "dur": 2,
                "args": {"rank": 0, "correlation": 7, "stream": 0},
            },
            {
                "ph": "f",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 7,
                "pid": 1,
                "tid": -3,
                "ts": 2,
                "args": {"rank": 0},
            },
        ],
    )

    summary = postprocess_profile(
        log_dir=log_dir,
        run_dir=run_dir,
        profile_kind="kernel",
        profile_step=1,
        trainer_log=trainer_log,
        nvidia_smi_csv=nvidia_smi,
    )

    assert summary["trace_file_count"] == 1
    assert summary["memory_snapshot_count"] == 1
    assert summary["peak_nvidia_smi_mib"] == 456
    assert summary["trainer_memory_gb"]["memory allocated"] == 1.25
    assert len(summary["archived_memory_snapshots"]) == 1
    assert (run_dir / "profile_summary.json").exists()
    assert (run_dir / "profile_summary.md").exists()
    assert (trace_dir / "traces-r0.gpu_only.chrome.json").exists()
    assert (
        run_dir / "kernel_traces" / "actor" / "traces-r0.gpu_only.chrome.json"
    ).exists()
    assert (
        run_dir / "memory_snapshots" / "step_1" / "snapshot_rank00_p0d0c0t0.pickle"
    ).exists()
