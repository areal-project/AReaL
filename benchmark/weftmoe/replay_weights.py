"""Replay AReaL colocated AWEX weight updates without rollout or training."""

import json
import os
import sys
import time

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("PerfTracer")


def main():
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    with PPOTrainer(config, train_dataset=dataset) as trainer:
        for version in range(1, int(os.environ.get("AWEX_REPLAY_STEPS", "5")) + 1):
            trainer.rollout.pause()
            trainer.rollout.pause_generation_sync()
            for tag in ("kv_cache", "weights", "cuda_graph"):
                trainer.rollout.offload(tags=[tag])
            trainer.actor.onload()
            started = time.perf_counter()
            trainer._update_weights_and_publish_version(
                trainer.weight_update_meta.with_version(version), version
            )
            elapsed = time.perf_counter() - started
            trainer.actor.offload()
            trainer._restore_awex_rollout_after_stats()
            row = dict(version=version, update_weights_s=elapsed)
            logger.info("AWEX_REPLAY_JSON %s", json.dumps(row))
            trainer.stats_logger.commit(
                0, version - 1, version - 1, {"weight_replay/update_s": elapsed}
            )


if __name__ == "__main__":
    main()
