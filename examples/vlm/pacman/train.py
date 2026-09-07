# SPDX-License-Identifier: Apache-2.0

"""Launch from the AReaL checkout; the original recipe checkout is a dependency."""

import hashlib
import json
import os
import sys
from pathlib import Path

from areal_pacman.level1.level1_dataset import make_episode_row, repository_revisions
from areal_pacman.level1.recipe import json_safe_value, recipe_contract_metadata
from omegaconf import OmegaConf

from examples.vlm.pacman.checkpoint import Checkpoint
from examples.vlm.pacman.config import PacmanConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


class RecipeInputs:
    RECIPE_REVISION = "219579e5323a0b24efb531b02f54ecc8848dd968"
    GAME_REVISION = "cbb97115e407abc86a44adc82a1b8f360b3e8da0"

    def __init__(self, config: PacmanConfig):
        if not os.environ.get("AREAL_ROOT") or not os.environ.get(
            "MAAPACMAN_PACMAN_ROOT"
        ):
            raise ValueError(
                "Export AREAL_ROOT and MAAPACMAN_PACMAN_ROOT on every worker"
            )
        if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0").lower() not in (
            "0",
            "false",
        ):
            raise ValueError(
                "SGLang must report the sampled constrained policy log-probability"
            )
        revisions = repository_revisions()
        for name, expected in (
            ("areal-pacman", self.RECIPE_REVISION),
            ("pacman-python", self.GAME_REVISION),
        ):
            if revisions[name]["commit"] != expected or revisions[name]["dirty"]:
                raise ValueError(f"{name} must be a clean checkout at {expected}")
        self.config, self.revisions = config, revisions

    def dataset(self, split: str) -> list[dict]:
        generation = self.config.dataset_generation
        count = (
            generation.train_episodes
            if split == "train"
            else generation.validation_episodes
        )
        seed = generation.seed + (0 if split == "train" else generation.train_episodes)
        return [
            make_episode_row(
                index + 1,
                split=split,
                seed=seed + index,
                max_steps=self.config.environment.max_steps,
                ghost_mode=self.config.environment.ghost_mode,
                action_protocol=self.config.action_protocol,
            )
            for index in range(count)
        ]

    def save_manifest(self, train: list[dict], valid: list[dict]) -> None:
        root = Path(self.config.artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        raw = OmegaConf.to_container(OmegaConf.structured(self.config), resolve=True)
        payload = {
            "source_revisions": self.revisions,
            "recipe_contract": recipe_contract_metadata(raw),
            "config": json_safe_value(raw),
            "train": train,
            "validation": valid,
            "dataset_kind": "regenerated release seeds; current AReaL provenance",
        }
        encoded = json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        (root / f"inputs-{digest}.json").write_bytes(encoded)


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PacmanConfig)
    if config.curriculum == 2:
        Checkpoint.validate(config.actor.path)
    inputs = RecipeInputs(config)
    train, valid = inputs.dataset("train"), inputs.dataset("validation")
    inputs.save_manifest(train, valid)
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        trainer.train(
            workflow="examples.vlm.pacman.workflow.PacmanWorkflow",
            workflow_kwargs=config.workflow_kwargs(config.gconfig, training=True),
            eval_workflow="examples.vlm.pacman.workflow.PacmanWorkflow",
            eval_workflow_kwargs=config.workflow_kwargs(
                config.eval_gconfig, training=False
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
