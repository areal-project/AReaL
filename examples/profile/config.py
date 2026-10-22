# SPDX-License-Identifier: Apache-2.0

"""Importable profile configuration for RPC worker serialization."""

from dataclasses import dataclass, field

from areal.api.cli_args import SFTConfig


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
