# SPDX-License-Identifier: Apache-2.0
"""Export the unmodified full Qwen3.8 Flash-Next checkpoint through Megatron-Bridge."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch.distributed as dist

from areal import SFTTrainer
from areal.api import SaveLoadMeta
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def _check_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"checkpoint identity changed: {path.name}: {digest}")


def main(args: list[str]) -> None:
    global_rank = int(os.environ.get("RANK", "0"))
    train_after_export = os.environ.get("TRAIN_AFTER_SOURCE_EXPORT", "0") == "1"
    if train_after_export:
        from examples.mtp.qwen38_flash_next.instrumented_sft import (
            install_step_instrumentation,
            install_structured_metrics,
        )

        install_step_instrumentation()
        install_structured_metrics()
    config, _ = load_expr_config(args, SFTConfig)
    model_root = Path(os.environ["MODEL_PATH"])
    _check_sha256(
        model_root / "config.json", os.environ["QWEN38_MTP_MODEL_CONFIG_SHA256"]
    )
    _check_sha256(
        model_root / "model.safetensors.index.json",
        os.environ["QWEN38_MTP_MODEL_INDEX_SHA256"],
    )

    trial = os.environ["SOURCE_ROUNDTRIP_TRIAL_NAME"]
    config.trial_name = trial
    config.actor.trial_name = trial
    config.saver.trial_name = trial
    config.evaluator.trial_name = trial
    config.stats_logger.trial_name = trial
    if config.recover.mode != "disabled" and not train_after_export:
        raise ValueError("source roundtrip must not recover trained weights")

    export_root = Path(os.environ["HF_EXPORT_ROOT"])
    if export_root.exists():
        raise FileExistsError(f"refusing to overwrite HF export: {export_root}")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )

    # SFTTrainer initializes the full Megatron model from the source HF weights.
    # Deliberately omit trainer.train(): every exported tensor must still equal
    # the source checkpoint, including the native MTP head.
    with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        if trainer.recover_info is not None:
            raise ValueError(
                "Source roundtrip requires pristine HF initialization, not recovery"
            )
        trainer.actor.save(
            SaveLoadMeta(
                path=str(export_root),
                weight_format="hf",
                with_optim=False,
                tokenizer=tokenizer,
                base_model_path=os.fspath(model_root),
            )
        )
        dist.barrier(group=trainer.actor.cpu_group)
        if train_after_export:
            trainer.train()

    if global_rank == 0:
        print(f"source_roundtrip_export PASS path={export_root}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
