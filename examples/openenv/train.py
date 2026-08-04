# SPDX-License-Identifier: Apache-2.0
"""Entry point for OpenEnv RL training with AReaL.

The workflow object is built inside ``main`` and passed as a class import path
plus kwargs to :class:`PPOTrainer.train`, so every episode consults the
configured environment (echo / blackjack / browsergym / ...).

The dataset is intentionally a thin in-memory seed generator: OpenEnv episodes
draw their prompt from ``env.reset()``, so per-example content is unnecessary.
For environments that require per-example inputs (e.g. a coding prompt), wire
a real dataset in and use the row content as the seed / task-id in
``data.get("seed", ...)``.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import OpenEnvExperimentConfig  # noqa: E402

from areal import PPOTrainer  # noqa: E402
from areal.api.cli_args import load_expr_config  # noqa: E402
from areal.utils.hf_utils import load_hf_tokenizer  # noqa: E402


class _SeedDataset:
    """Iterable dataset yielding ``{"seed": int}`` rows.

    Kept private to this example: production workflows should back their
    dataset with a real prompt source rather than a numeric seed.
    """

    def __init__(self, size: int, offset: int = 0) -> None:
        self._size = size
        self._offset = offset

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"seed": self._offset + idx}


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, OpenEnvExperimentConfig)
    _tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = _SeedDataset(size=config.dataset_size)
    valid_dataset = _SeedDataset(size=config.eval_dataset_size, offset=10_000_000)

    workflow_kwargs = dict(
        config=config.openenv,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        initial_user_prompt=config.initial_user_prompt,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.openenv.OpenEnvWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.openenv.OpenEnvWorkflow",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
