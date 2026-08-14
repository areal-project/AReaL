"""MemRL command-line interface.

The primary way to run benchmarks in this repo is via the scripts in `run/`.
This CLI exists mainly for packaging sanity and basic introspection.
"""

from __future__ import annotations

import click

from memrl.__version__ import __description__, __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="memrl")
def cli() -> None:
    """MemRL utilities."""


@cli.command("about")
def about_cmd() -> None:
    """Print a short project description."""
    click.echo(__description__)

