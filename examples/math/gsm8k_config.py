# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class GSM8KGRPOConfig(GRPOConfig):
    reward_max_workers: int | None = field(
        default=None,
        metadata={"help": "Maximum worker processes used to compute math rewards."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.reward_max_workers is not None and self.reward_max_workers <= 0:
            raise ValueError(
                "reward_max_workers must be positive when set, got "
                f"{self.reward_max_workers}"
            )
