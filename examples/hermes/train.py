"""Online training: external users interact via proxy gateway URL."""

import argparse
import json
import sys
from pathlib import Path

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--elastic-revision", action="store_true")
    parser.add_argument("--elastic-profile", type=Path)
    parser.add_argument("--elastic-settings", type=Path)
    parser.add_argument("--elastic-bundle", type=Path)
    options, args = parser.parse_known_args(args)
    config, _ = load_expr_config(args, PPOConfig)
    kwargs = {}
    if options.elastic_revision or any(
        (options.elastic_profile, options.elastic_settings, options.elastic_bundle)
    ):
        from areal.elastic.infra.adapter import ReconfiguratorAdapter

        adapter_type = ReconfiguratorAdapter
        adapter_options = {}
        if options.elastic_revision or options.elastic_bundle is not None:
            from areal.autopilot.domains.elastic.native import RevisionAdapter

            adapter_type = RevisionAdapter
            adapter_options["bundle"] = options.elastic_bundle
        kwargs["elastic_reconfigurator"] = adapter_type(
            profile=None
            if options.elastic_profile is None
            else json.loads(options.elastic_profile.read_text()),
            settings=None
            if options.elastic_settings is None
            else json.loads(options.elastic_settings.read_text()),
            **adapter_options,
        )

    with PPOTrainer(config, **kwargs) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
