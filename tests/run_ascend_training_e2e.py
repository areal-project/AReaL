# SPDX-License-Identifier: Apache-2.0

"""Run the real two-step Geometry3K trainer and record completion evidence."""

import faulthandler
import json
import sys
from pathlib import Path
from unittest.mock import patch

from safetensors import safe_open

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.saver import Saver


def main(args: list[str]) -> None:
    faulthandler.dump_traceback_later(600, repeat=True)
    config, _ = load_expr_config(args, GRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train[:2]",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = get_custom_dataset(
        split="test[:1]",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    assert len(train_dataset) == 2
    assert len(valid_dataset) == 1
    workflow_kwargs = dict(
        reward_fn="examples.vlm.geometry3k_grpo.geometry3k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_kwargs = {**workflow_kwargs, "gconfig": config.eval_gconfig}
    # Hardware telemetry is outside this training E2E and may block in DCMI init.
    with (
        patch(
            "areal.utils.stats_logger.install_dcmi_ascend_collector",
            return_value="unavailable",
        ),
        PPOTrainer(
            config, train_dataset=train_dataset, valid_dataset=valid_dataset
        ) as trainer,
    ):
        # These spies call through to the real implementations.
        with (
            patch.object(
                trainer.stats_logger, "commit", wraps=trainer.stats_logger.commit
            ) as commits,
            patch.object(
                trainer, "_evaluate_fn", wraps=trainer._evaluate_fn
            ) as evaluations,
        ):
            trainer.train(
                workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
                eval_workflow_kwargs=eval_kwargs,
            )
            steps = [call.args[2] for call in commits.call_args_list]
            assert steps == [0, 1], steps
            assert evaluations.call_count == 1, evaluations.call_count

    checkpoint = Path(
        Saver.get_model_save_path(
            config.experiment_name,
            config.trial_name,
            config.cluster.fileroot,
            epoch=0,
            step=1,
            globalstep=1,
        )
    )
    assert (checkpoint / "config.json").is_file(), checkpoint
    assert (checkpoint / "tokenizer_config.json").is_file(), checkpoint
    weights = sorted(checkpoint.glob("*.safetensors"))
    assert weights, checkpoint
    keys = set()
    for weight in weights:
        with safe_open(weight, framework="pt", device="cpu") as handle:
            shard_keys = list(handle.keys())
            assert shard_keys, weight
            keys.update(shard_keys)
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        assert set(weight_map).issubset(keys)
        assert set(weight_map.values()).issubset({path.name for path in weights})
    evidence = {
        "completed_train_steps": len(steps),
        "completed_evaluations": evaluations.call_count,
        "evaluation_examples": len(valid_dataset),
        "checkpoint": str(checkpoint),
        "weight_files": [path.name for path in weights],
        "saved_tensor_count": len(keys),
    }
    (Path(config.cluster.fileroot) / "completion.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main(sys.argv[1:])
