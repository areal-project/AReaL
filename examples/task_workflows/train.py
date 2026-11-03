"""Shared online PPO entry point for externally submitted benchmark tasks."""

import argparse
import json
import sys
from pathlib import Path

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--elastic-profile", type=Path)
    parser.add_argument("--elastic-settings", type=Path)
    options, args = parser.parse_known_args(args)
    config, _ = load_expr_config(args, PPOConfig)
    kwargs = {}
    if options.elastic_profile is not None or options.elastic_settings is not None:
        from areal.elastic.infra.adapter import ReconfiguratorAdapter

        kwargs["elastic_reconfigurator"] = ReconfiguratorAdapter(
            profile=None
            if options.elastic_profile is None
            else json.loads(options.elastic_profile.read_text()),
            settings=None
            if options.elastic_settings is None
            else json.loads(options.elastic_settings.read_text()),
        )

    with PPOTrainer(config, **kwargs) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
