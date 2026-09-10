# SPDX-License-Identifier: Apache-2.0

"""SFT profile entrypoint with deterministic fake 128K samples."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger

logger = getLogger("ProfileSFT")


_PROFILE_FAKE_TEXT = """
System: You are an expert software engineer working inside a large Python
repository. Preserve public APIs, keep the patch local, and explain the exact
validation that proves the fix.

User: The 128K-token SFT profile run regressed after switching the MoE layout
from a pure data-parallel baseline to context-parallel plus expert-parallel
training. The symptom is a mismatch between kernel time, loss-mask coverage,
and peak CUDA allocator memory. Please inspect the fake training sample,
identify the smallest reliable fix, and write the validation plan.

Assistant: I will make the profile input deterministic and realistic enough to
exercise long-context attention, tokenizer boundaries, code blocks, tool-call
style JSON, and a dense assistant target span.

Context:
- The profile sample must be identical across baseline and alternative
  parallel layouts.
- The batch loader must not depend on external JSONL files.
- The loss mask should exclude prompt/context tokens and include the assistant
  target tokens.
- Kernel profile and memory profile should be separable because torch profiler
  memory recording can perturb the kernel timeline.

Repository excerpt:
```python
def normalize_batch(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    input_ids = sample["input_ids"]
    loss_mask = sample["loss_mask"]
    if input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids and loss_mask must have identical shapes")
    if loss_mask.count_nonzero() == 0:
        raise ValueError("profile sample must contain trainable target tokens")
    return {"input_ids": input_ids, "loss_mask": loss_mask}
```

Tool result:
{"status": "failed", "rank": 3, "step": 1, "reason": "trace missing GPU kernels"}

Patch plan:
1. Gate PyTorch profiler execution on perf_tracer.profile_steps.
2. Keep CUDA memory snapshots controlled by memory_profiler.profile_steps.
3. Select ranks through AREAL_PERF_TRACER_RANKS and AREAL_MEMORY_PROFILER_RANKS.
4. Convert PerfTracer JSONL to Chrome trace views split by CPU, CUDA API, and GPU
   kernels.
5. Archive summary files next to the raw trace and snapshot artifacts.

Final response: The profile run is deterministic. The fake 128K sequence is a
repeated structured conversation with code, JSON, and validation text. The
second half of the sequence is included in the SFT loss mask so every layout
optimizes the same target span.
"""


@dataclass
class ProfileDataConfig:
    """Configuration for deterministic fake SFT profile data."""

    fake_seq_len: int = field(
        default=131072,
        metadata={"help": "Token length of each fake profile sample."},
    )
    fake_dataset_size: int = field(
        default=8,
        metadata={"help": "Logical dataset size for repeated fake samples."},
    )
    fake_loss_start_ratio: float = field(
        default=0.5,
        metadata={
            "help": "Fraction of each sequence treated as prompt/context tokens."
        },
    )


@dataclass
class ProfileSFTConfig(SFTConfig):
    """SFT config extended with profile fake-data options."""

    profile: ProfileDataConfig = field(default_factory=ProfileDataConfig)


class RepeatedProfileSFTDataset(Dataset):
    """Deterministic repeated long-context samples for profile comparisons."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        seq_len: int,
        dataset_size: int,
        loss_start_ratio: float,
    ) -> None:
        if seq_len < 2:
            raise ValueError("profile.fake_seq_len must be at least 2.")
        if dataset_size < 1:
            raise ValueError("profile.fake_dataset_size must be at least 1.")
        if not 0.0 <= loss_start_ratio < 1.0:
            raise ValueError("profile.fake_loss_start_ratio must be in [0.0, 1.0).")

        base_ids = tokenizer.encode(_PROFILE_FAKE_TEXT, add_special_tokens=False)
        if not base_ids:
            fallback_id = tokenizer.eos_token_id
            if fallback_id is None:
                fallback_id = tokenizer.pad_token_id
            if fallback_id is None:
                fallback_id = 0
            base_ids = [fallback_id]

        repeats = seq_len // len(base_ids) + 1
        input_ids = (base_ids * repeats)[:seq_len]
        loss_start = min(seq_len - 1, max(1, int(seq_len * loss_start_ratio)))
        loss_mask = [0] * loss_start + [1] * (seq_len - loss_start)

        self._input_ids = torch.tensor(input_ids, dtype=torch.long)
        self._loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        self._dataset_size = dataset_size

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        return {
            "input_ids": self._input_ids.clone(),
            "loss_mask": self._loss_mask.clone(),
        }


def build_profile_dataset(config: ProfileSFTConfig) -> RepeatedProfileSFTDataset:
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = RepeatedProfileSFTDataset(
        tokenizer,
        seq_len=config.profile.fake_seq_len,
        dataset_size=config.profile.fake_dataset_size,
        loss_start_ratio=config.profile.fake_loss_start_ratio,
    )
    logger.info(
        "Using fake profile SFT data: seq_len=%s, dataset_size=%s, loss_start_ratio=%s",
        config.profile.fake_seq_len,
        config.profile.fake_dataset_size,
        config.profile.fake_loss_start_ratio,
    )
    return dataset


def main(args: list[str]) -> None:
    from areal import SFTTrainer

    config, _ = load_expr_config(args, ProfileSFTConfig)
    train_dataset = build_profile_dataset(config)

    with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
