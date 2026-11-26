# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from areal.api.cli_args import DatasetSourceConfig, _DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from areal.infra.data_service.rdataset import RDataset

VALID_DATASETS = [
    "gsm8k",
    "clevr_count_70k",
    "geometry3k",
    "virl39k",
    "hh-rlhf",
    "torl_data",
    "aime",
    "leetcode",
    "decision",
]

logger = logging.getLogger("Dataset")


def _is_decision_path(path: str | None) -> bool:
    from .decision import is_decision_path

    return is_decision_path(path)


def _add_or_replace_column(dataset: "Dataset", name: str, value: str) -> "Dataset":
    if name in dataset.column_names:
        dataset = dataset.remove_columns([name])
    return dataset.add_column(name, [value] * len(dataset))


def _concatenate_tagged_datasets(datasets: list["Dataset"]) -> "Dataset":
    from datasets import Dataset, concatenate_datasets

    try:
        return concatenate_datasets(datasets)
    except Exception as concat_err:
        logger.warning(
            "Falling back to row-wise mixed dataset construction because "
            "concatenate_datasets failed: %s",
            concat_err,
        )
        rows: list[dict[str, Any]] = []
        all_columns: set[str] = set()
        for dataset in datasets:
            all_columns.update(dataset.column_names)
            rows.extend(dict(row) for row in dataset)

        for row in rows:
            for column in all_columns:
                row.setdefault(column, None)

        return Dataset.from_list(rows)


def _repeat_dataset_to_length(dataset: "Dataset", target_size: int) -> "Dataset":
    if len(dataset) == 0:
        raise ValueError("Cannot equalize an empty dataset source.")
    if len(dataset) == target_size:
        return dataset

    repeats, remainder = divmod(target_size, len(dataset))
    parts = [dataset] * repeats
    if remainder:
        parts.append(dataset.shuffle(seed=0).select(range(remainder)))
    return _concatenate_tagged_datasets(parts)


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if type == "decision" or (type == "sft" and _is_decision_path(path)):
        from .decision import get_decision_sft_dataset

        return get_decision_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            processor=processor,
            **kwargs,
        )
    if "gsm8k" in path and type == "sft":
        from .gsm8k import get_gsm8k_sft_dataset

        return get_gsm8k_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "gsm8k" in path and type == "rl":
        from .gsm8k import get_gsm8k_rl_dataset

        return get_gsm8k_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "aime" in path and type == "sft":
        from .aime import get_aime_sft_dataset

        return get_aime_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "aime" in path and type == "rl":
        from .aime import get_aime_rl_dataset

        return get_aime_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "leetcode" in path and type == "rl":
        from .leetcode import get_leetcode_rl_dataset

        return get_leetcode_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "sft":
        from .clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "rl":
        from .clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "sft":
        from .geometry3k import get_geometry3k_sft_dataset

        return get_geometry3k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "rl":
        from .geometry3k import get_geometry3k_rl_dataset

        return get_geometry3k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "virl39k" in path.lower() and type == "rl":
        from .virl39k import get_virl39k_rl_dataset

        return get_virl39k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "hh-rlhf" in path and type == "rw":
        from .hhrlhf import get_hhrlhf_rw_dataset

        return get_hhrlhf_rw_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "hh-rlhf" in path and type == "dpo":
        from .hhrlhf import get_hhrlhf_dpo_dataset

        return get_hhrlhf_dpo_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "torl_data" in path and type == "rl":
        from .torl_data import get_torl_data_rl_dataset

        return get_torl_data_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    else:
        # Fallback: try loading as a generic HuggingFace dataset from disk.
        # This supports arbitrary datasets saved via dataset.save_to_disk().
        try:
            from datasets import DatasetDict, load_from_disk

            dataset = load_from_disk(path)
            if isinstance(dataset, DatasetDict):
                if split is not None:
                    if split in dataset:
                        return dataset[split]
                    available = list(dataset.keys())
                    raise ValueError(
                        f"Requested split '{split}' not found in DatasetDict at {path}. "
                        f"Available splits: {available}"
                    )
                available = list(dataset.keys())
                if available:
                    return dataset[available[0]]
                raise ValueError(f"Empty DatasetDict at {path}")
            return dataset
        except Exception as load_err:
            raise ValueError(
                f"Dataset {path} with split {split} and training type {type} is not supported. "
                f"Supported datasets are: {VALID_DATASETS}. "
                f"Also failed to load from disk: {load_err}"
            )


def _get_mixed_dataset(
    sources: list[DatasetSourceConfig],
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    upsample_to_largest: bool = False,
    **kwargs,
) -> "Dataset":
    if len(sources) == 0:
        raise ValueError("At least one dataset source is required for mixed datasets.")

    datasets = []
    for source in sources:
        source_kwargs: dict[str, Any] = {
            **kwargs,
            **(source.dataset_kwargs or {}),
        }
        dataset = _get_custom_dataset(
            path=source.path,
            type=source.type,
            split=source.split if source.split is not None else split,
            max_length=(
                source.max_length if source.max_length is not None else max_length
            ),
            tokenizer=tokenizer,
            processor=processor,
            **source_kwargs,
        )
        dataset = _add_or_replace_column(dataset, "domain", source.domain)
        dataset = _add_or_replace_column(dataset, "dataset_name", source.name)
        datasets.append(dataset)

    if upsample_to_largest:
        empty_sources = [
            source.name
            for source, dataset in zip(sources, datasets)
            if len(dataset) == 0
        ]
        if empty_sources:
            raise ValueError(
                "Cannot equalize empty dataset sources: " + ", ".join(empty_sources)
            )
        target_size = max(len(dataset) for dataset in datasets)
        datasets = [
            _repeat_dataset_to_length(dataset, target_size) for dataset in datasets
        ]

    return _concatenate_tagged_datasets(datasets)


def get_custom_dataset(
    split: str | None = None,
    dataset_config: _DatasetConfig | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset | RDataset":
    from areal.utils.environ import is_single_controller

    if (
        is_single_controller()
        and dataset_config is not None
        and dataset_config.scheduling_spec is not None
    ):
        if len(dataset_config.datasets) > 0:
            raise ValueError(
                "Remote RDataset loading does not support train_dataset.datasets "
                "yet. Set train_dataset.scheduling_spec=null to use local mixed "
                "dataset loading."
            )
        from areal.infra.data_service.rdataset import RDataset

        if dataset_config.path is None or dataset_config.type is None:
            raise ValueError(
                "dataset_config.path and dataset_config.type are required when "
                "dataset_config.datasets is empty."
            )

        return RDataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split if split is not None else dataset_config.split,
            max_length=dataset_config.max_length,
            dataset_kwargs=getattr(dataset_config, "dataset_kwargs", None),
        )

    if dataset_config is not None:
        effective_split = split if split is not None else dataset_config.split
        dataset_kwargs: dict[str, Any] = {
            **getattr(dataset_config, "dataset_kwargs", {}),
            **kwargs,
        }
        if len(dataset_config.datasets) > 0:
            return _get_mixed_dataset(
                sources=dataset_config.datasets,
                split=effective_split,
                max_length=dataset_config.max_length,
                tokenizer=tokenizer,
                processor=processor,
                upsample_to_largest=dataset_config.upsample_to_largest,
                **dataset_kwargs,
            )

        if dataset_config.path is None or dataset_config.type is None:
            raise ValueError(
                "dataset_config.path and dataset_config.type are required when "
                "dataset_config.datasets is empty."
            )

        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=effective_split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **dataset_kwargs,
        )

    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )


__all__ = [
    "VALID_DATASETS",
    "get_custom_dataset",
]
