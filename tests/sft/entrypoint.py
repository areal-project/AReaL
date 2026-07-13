import json
import os
import sys

import torch.distributed as dist

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.data_service.rdataset import RDataset
from areal.utils.hf_utils import load_hf_tokenizer


class MinimalSFTTrainer(SFTTrainer):
    """Trainer that collects stats to JSON for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses_history = []

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # Collect stats before committing
        stats = self.actor.export_stats()
        self.losses_history.append(stats["sft/loss/avg"])


def main() -> None:
    os.environ["AREAL_SPMD_MODE"] = "0"

    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    assert isinstance(train_dataset, RDataset), (
        "SFT integration test expects get_custom_dataset() to return RDataset "
        "(single-controller data service path)."
    )

    with MinimalSFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer.train()

        # Save losses to JSON for test assertions
        if dist.get_rank() == 0:
            with open(os.path.join(config.cluster.fileroot, "losses.json"), "w") as f:
                json.dump(trainer.losses_history, f)


if __name__ == "__main__":
    main()
