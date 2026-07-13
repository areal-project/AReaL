# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

from areal.api.cli_args import _DatasetConfig
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
]

logger = logging.getLogger("Dataset")


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
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
        from areal.infra.data_service.rdataset import RDataset

        return RDataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            dataset_kwargs=getattr(dataset_config, "dataset_kwargs", None),
        )

    if dataset_config is not None:
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
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
