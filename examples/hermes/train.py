"""Online training: external users interact via proxy gateway URL."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args):
    config, _ = load_expr_config(args, PPOConfig)

    with PPOTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
