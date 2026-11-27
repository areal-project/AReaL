"""Train Pixel visual-tool reasoning with GRPO and AReaL's PPOTrainer."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config


@dataclass
class PixelReasonerConfig(GRPOConfig):
    data_root: str = ""
    max_turns: int | None = None
    max_output_tokens: int | None = None
    original_max_pixels: int = 1568000
    crop_max_pixels: int = 784000


def load_manifest(path: str, data_root: str) -> Dataset:
    """Load a prepared, labelled JSONL split without embedding image bytes."""
    root = Path(data_root).resolve()
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Every task needs a nonempty question")
            if not isinstance(row.get("image"), list) or not row["image"]:
                raise ValueError("Every task needs a nonempty image/frame list")
            answers = row.get("answer")
            if (
                not isinstance(answers, list)
                or len(answers) != 1
                or not isinstance(answers[0], str)
                or not answers[0].strip()
            ):
                raise ValueError(
                    "Exact-match training requires one nonempty reference answer"
                )
            if type(row.get("is_video", False)) is not bool:
                raise ValueError("is_video must be a boolean")
            for name in row["image"]:
                if not isinstance(name, str) or not name or Path(name).is_absolute():
                    raise ValueError("Visual paths must be nonempty relative strings")
                image = (root / name).resolve()
                if not image.is_relative_to(root) or not image.is_file():
                    raise ValueError(
                        f"Visual input must be an existing file under data_root: {name}"
                    )
            rows.append(
                dict(
                    question=row["question"],
                    image=row["image"],
                    answer=answers,
                    is_video=bool(row.get("is_video", False)),
                )
            )
    if not rows:
        raise ValueError(f"Empty Pixel manifest: {path}")
    return Dataset.from_list(rows)


def main(args: list[str]):
    config, _ = load_expr_config(args, PixelReasonerConfig)
    if config.rollout.backend.split(":", 1)[0] != "sglang":
        raise ValueError("The Pixel training example requires the SGLang backend")
    if config.actor.backend.split(":", 1)[0] != "megatron":
        raise ValueError("The Qwen3.8-27B Pixel example requires the Megatron backend")
    if not config.data_root:
        raise ValueError("Set data_root to the shared image/frame bundle")

    train_dataset = load_manifest(config.train_dataset.path, config.data_root)
    valid_dataset = load_manifest(config.valid_dataset.path, config.data_root)
    workflow_kwargs = dict(
        processor=config.tokenizer_path,
        gconfig=config.gconfig,
        data_root=config.data_root,
        max_turns=config.max_turns,
        max_output_tokens=config.max_output_tokens,
        original_max_pixels=config.original_max_pixels,
        crop_max_pixels=config.crop_max_pixels,
        video_temp_root=str(
            Path(config.cluster.fileroot)
            / config.experiment_name
            / config.trial_name
            / "pixel-video"
        ),
    )
    eval_workflow_kwargs = {**workflow_kwargs, "gconfig": config.eval_gconfig}
    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train(
            workflow="examples.vlm.pixel_reasoner_workflow.PixelReasonerWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.vlm.pixel_reasoner_workflow.PixelReasonerWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    # RPC workers need the config class under its importable module name.
    from examples.vlm.pixel_reasoner import main as canonical_main

    canonical_main(sys.argv[1:])
