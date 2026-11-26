# SPDX-License-Identifier: Apache-2.0

"""Train pure MOPD on freshly generated continuations of offline prefixes."""

from __future__ import annotations

import getpass
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from examples.prefix_replay.config import (
    PrefixReplayOPDConfig,
    build_prefix_replay_workflow_kwargs,
)

from areal.api.cli_args import load_expr_config
from areal.dataset.mopd import get_mopd_dataset
from areal.dataset.prefix_replay import (
    PrefixReplayIndexedDataset,
    load_prefix_replay_indexed_dataset,
)
from areal.dataset.prefix_replay_cache import (
    ProcessedPrefixReplayCacheLock,
    build_prefix_replay_cache_metadata,
    preflight_prefix_replay_cache,
    read_prefix_replay_cache,
    write_prefix_replay_cache,
)
from areal.utils.hf_utils import load_hf_tokenizer


def select_route(
    dataset: PrefixReplayIndexedDataset, field: str, route: str
) -> PrefixReplayIndexedDataset:
    """Select a source's route without expanding its shared trajectory storage."""
    indices = [
        index
        for index in dataset.indices
        if dataset.trajectories[index["trajectory_index"]].get(field) == route
    ]
    if not indices:
        raise ValueError(f"Replay source has no prefixes for {field}={route!r}")
    return PrefixReplayIndexedDataset(dataset.trajectories, indices)


def load_replay_dataset(
    config: PrefixReplayOPDConfig,
    tokenizer: Any,
    *,
    split: str,
    stack: ExitStack,
    cache_dirs: set[Path],
):
    """Load source-routed compact datasets, retaining cache leases during training.

    Set source.dataset_kwargs.route_field and route_value to select one route
    from an already mixed replay file. Teacher selection remains source-owned.
    """
    dataset_config = config.train_dataset if split == "train" else config.valid_dataset
    if dataset_config is None:
        return None
    options = config.prefix_replay
    source_index = 0
    route_coverage: dict[tuple[str, str], tuple[set[str], set[str]]] = {}

    def load_source(*, source_config: Any, **kwargs: Any):
        nonlocal source_index
        cache_dir = (
            Path(config.cluster.fileroot)
            / "checkpoints"
            / getpass.getuser()
            / config.experiment_name
            / config.trial_name
            / f"processed_prefix_replay_{split}_{source_index}"
        )
        source_index += 1
        limits = [config.gconfig.max_tokens]
        if config.rollout.agent.engine_max_tokens is not None:
            limits.append(config.rollout.agent.engine_max_tokens)
        if config.sglang.context_length is not None:
            limits.append(config.sglang.context_length)
        max_length = min(limits) - 1
        if max_length < 1:
            raise ValueError("Replay requires room for a prefix and a student action")
        if source_config.max_length is not None:
            max_length = min(max_length, source_config.max_length)
        source_options = dict(source_config.dataset_kwargs)
        route_field = source_options.pop("route_field", None)
        route_value = source_options.pop("route_value", None)
        if source_options:
            raise ValueError(
                f"Unknown prefix replay dataset options: {sorted(source_options)}"
            )
        if (route_field is None) != (route_value is None):
            raise ValueError("route_field and route_value must be specified together")
        if route_field is not None and (
            not isinstance(route_field, str)
            or not route_field.strip()
            or not isinstance(route_value, str)
            or not route_value.strip()
        ):
            raise ValueError("route_field and route_value must be non-empty strings")
        loader_options = dict(
            kappa=options.kappa,
            seed=options.seed,
            input_mode=options.input_mode,
            drop_system_messages=options.drop_system_messages,
            route_field=route_field,
            route_metadata_field=options.route_metadata_field,
            route_default_value=options.default_route,
            parse_tool_call_args=options.parse_tool_call_args,
            max_length=max_length,
            chat_template_kwargs=dict(config.rollout.agent.chat_template_kwargs),
        )
        cache_meta = build_prefix_replay_cache_metadata(
            source_config.path,
            split=split,
            tokenizer_path=config.tokenizer_path,
            **loader_options,
        )
        use_cache = (
            options.cache_processed_dataset
            and os.getenv("PREFIX_REPLAY_DISABLE_CACHE", "0") != "1"
        )
        rows = None
        if use_cache:
            lock = ProcessedPrefixReplayCacheLock(cache_dir).acquire()
            stack.callback(lock.close)
            cache_dirs.add(cache_dir)
            cache_valid, _ = preflight_prefix_replay_cache(cache_dir, cache_meta)
            if cache_valid:
                rows = read_prefix_replay_cache(cache_dir, cache_meta)
        if rows is None:
            rows = load_prefix_replay_indexed_dataset(
                source_config.path, tokenizer=tokenizer, **loader_options
            )
            if use_cache:
                write_prefix_replay_cache(cache_dir, cache_meta, rows)
        if route_field is not None:
            key = (str(Path(source_config.path).resolve()), route_field)
            observed, selected = route_coverage.setdefault(key, (set(), set()))
            if route_value in selected:
                raise ValueError(
                    f"Duplicate replay route selection: {key}={route_value!r}"
                )
            if rows.first_missing_route_index(route_field) is not None:
                raise ValueError(
                    f"Replay source is missing route field {route_field!r}"
                )
            observed.update(rows.route_values(route_field))
            selected.add(route_value)
            rows = select_route(rows, route_field, route_value)
        return rows

    dataset = get_mopd_dataset(
        dataset_config, tokenizer=tokenizer, source_loader=load_source
    )
    for (path, field), (observed, selected) in route_coverage.items():
        if missing := observed - selected:
            raise ValueError(
                f"Unconfigured replay routes in {path} ({field}): {sorted(missing)}"
            )
    return dataset


def main(args: list[str]) -> None:
    from areal import PPOTrainer

    config, _ = load_expr_config(args, PrefixReplayOPDConfig)
    if config.mopd is None:
        raise ValueError("Prefix replay requires MOPD, including for a single teacher")
    if config.mopd.loss.rl_coefficient != 0.0:
        raise ValueError(
            "Prefix replay has zero environment reward; set rl_coefficient=0"
        )
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    cache_dirs: set[Path] = set()
    with ExitStack() as stack:
        train_dataset = load_replay_dataset(
            config, tokenizer, split="train", stack=stack, cache_dirs=cache_dirs
        )
        valid_dataset = load_replay_dataset(
            config, tokenizer, split="valid", stack=stack, cache_dirs=cache_dirs
        )
        kwargs = build_prefix_replay_workflow_kwargs(config.gconfig)
        total_limit = min(
            value
            for value in (
                config.gconfig.max_tokens,
                config.rollout.agent.engine_max_tokens,
                config.sglang.context_length,
            )
            if value is not None
        )
        kwargs["extra_body"]["max_total_tokens"] = total_limit
        eval_kwargs = build_prefix_replay_workflow_kwargs(
            config.eval_gconfig or config.gconfig
        )
        with PPOTrainer(
            config, train_dataset=train_dataset, valid_dataset=valid_dataset
        ) as trainer:
            trainer.train(
                workflow="examples.prefix_replay.agent.PrefixReplayAgent",
                workflow_kwargs=kwargs,
                eval_workflow=None,
                eval_workflow_kwargs=eval_kwargs,
                dynamic_filter_fn=None,
            )
        if config.prefix_replay.cleanup_processed_dataset:
            for cache_dir in cache_dirs:
                shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
