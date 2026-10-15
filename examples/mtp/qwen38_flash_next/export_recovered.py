# SPDX-License-Identifier: Apache-2.0
"""Recover the MTP-only run, execute its next step, and export HF weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

from examples.mtp.qwen38_flash_next.instrumented_sft import (
    install_step_instrumentation,
    install_structured_metrics,
)

from areal import SFTTrainer
from areal.api import SaveLoadMeta
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    install_step_instrumentation()
    install_structured_metrics()
    config, _ = load_expr_config(args, SFTConfig)

    # Recover from the original interrupted trial, while keeping this export
    # rerun's logs and ancillary state separate.  The explicit engine.save
    # below intentionally does not depend on Saver frequency state restored
    # from the DCP checkpoint.
    export_trial = os.environ["EXPORT_TRIAL_NAME"]
    config.trial_name = export_trial
    config.actor.trial_name = export_trial
    config.saver.trial_name = export_trial
    config.evaluator.trial_name = export_trial
    config.stats_logger.trial_name = export_trial

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    export_root = Path(os.environ["HF_EXPORT_ROOT"])
    if export_root.exists():
        raise FileExistsError(f"refusing to overwrite HF export: {export_root}")

    with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        from examples.mtp.qwen38_flash_next.recovery_checks import (
            snapshot_training_state,
        )

        snapshot_training_state(trainer.actor, "restored-boundary")
        trainer.train()
        trainer.actor.save(
            SaveLoadMeta(
                path=str(export_root),
                weight_format="hf",
                with_optim=False,
                tokenizer=tokenizer,
                base_model_path=os.fspath(config.actor.path),
            )
        )
        dist.barrier(group=trainer.actor.cpu_group)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"export_recovered PASS path={export_root}")


if __name__ == "__main__":
    main(sys.argv[1:])
