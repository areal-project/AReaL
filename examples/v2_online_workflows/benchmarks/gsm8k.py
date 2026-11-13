# SPDX-License-Identifier: Apache-2.0
"""Frozen GSM8K splits with the original example's prompt and answer grading."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.types import (
    EnvironmentSpec,
    GradeResult,
    TaskInput,
    TaskResult,
)

DATASET = "openai/gsm8k"
PROMPT_SUFFIX = "\nPlease put your final answer within \\boxed{}."


def question_id(question: str, split: str, index: int) -> str:
    return f"gsm8k-{split}-{index:05d}-" + hashlib.sha256(question.encode()).hexdigest()


@dataclass(frozen=True)
class GSM8KTask:
    task_id: str
    split: str
    source_index: int
    question: str
    answer: str


def tasks_from_rows(rows: Iterable[dict[str, Any]], split: str) -> list[GSM8KTask]:
    if split not in ("train", "test"):
        raise ValueError("Use the official GSM8K train or test split")
    tasks = []
    for index, row in enumerate(rows):
        question, answer = row["question"], row["answer"]
        if not all(
            isinstance(value, str) and value.strip() for value in (question, answer)
        ):
            raise ValueError(
                "GSM8K requires nonempty question and complete gold answer"
            )
        tasks.append(
            GSM8KTask(
                question_id(question, split, index), split, index, question, answer
            )
        )
    return tasks


def freeze_splits(
    splits: dict[str, Iterable[dict[str, Any]]], output: Path, *, source: dict[str, Any]
) -> dict[str, Any]:
    """Preserve upstream order, duplicates and complete labels; no length filter."""
    if set(splits) != {"train", "test"}:
        raise ValueError("Freeze official train and test as separate task files")
    tasks = {
        split: tasks_from_rows(splits[split], split) for split in ("train", "test")
    }
    if not all(tasks.values()):
        raise ValueError("GSM8K splits must be nonempty")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {
        "format_version": 1,
        "dataset": DATASET,
        "config": "main",
        "source": source,
        "splits": {},
    }
    for split, rows in tasks.items():
        payload = "".join(
            json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in rows
        ).encode()
        path = output / f"{split}.jsonl"
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o600)
        manifest["splits"][path.name] = {
            "split": split,
            "count": len(rows),
            "training_allowed": split == "train",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    write_json(output / "manifest.json", manifest)
    return manifest


def add_original_sampling_order(
    output: Path, *, epochs: int, batch_size: int
) -> dict[str, Any]:
    """Freeze the original controller's DistributedSampler order, not batch selection."""
    from torch.utils.data import DistributedSampler

    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    count = manifest["splits"]["train.jsonl"]["count"]
    if epochs < 1 or batch_size < 1:
        raise ValueError("Explicit positive epoch and batch settings are required")
    # TrainController exposes world_size=1/rank=0 to its coordinator dataloader.
    # areal.utils.dataloader does not override DistributedSampler's seed=0.
    sampler = DistributedSampler(
        range(count), num_replicas=1, rank=0, shuffle=True, drop_last=True
    )
    per_epoch = count // batch_size * batch_size
    indices = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        indices.extend(list(sampler)[:per_epoch])
    order = {
        "kind": "original_DistributedSampler",
        "sampler_seed": sampler.seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "drop_last": True,
        "source_count": count,
        "samples_per_epoch": per_epoch,
        "indices": indices,
    }
    order_path = output / "train-order.json"
    write_json(order_path, order)
    record = {k: v for k, v in order.items() if k != "indices"}
    record.update(
        file=order_path.name,
        sha256=hashlib.sha256(order_path.read_bytes()).hexdigest(),
        entries=len(indices),
    )
    manifest["splits"]["train.jsonl"]["sampling_order"] = record
    write_json(path, manifest)
    return record


class GSM8KBenchmark:
    grade_in_process = True
    name = "gsm8k"
    request_prefix = "gsm8k"

    @staticmethod
    def restore_task(value: dict[str, Any]) -> GSM8KTask:
        return GSM8KTask(**value)

    @staticmethod
    def load_tasks(path: Path) -> tuple[list[GSM8KTask], dict[str, Any]]:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        entry = manifest["splits"][path.name]
        payload = path.read_bytes()
        if (
            manifest["format_version"] != 1
            or hashlib.sha256(payload).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Frozen GSM8K task file changed")
        tasks = [
            GSM8KTask(**json.loads(line))
            for line in payload.splitlines()
            if line.strip()
        ]
        if len(tasks) != entry["count"]:
            raise ValueError("Frozen GSM8K count changed")
        for index, task in enumerate(tasks):
            if (
                task.split != entry["split"]
                or task.source_index != index
                or task.task_id != question_id(task.question, task.split, index)
            ):
                raise ValueError("Frozen GSM8K split or identity changed")
        source_count = len(tasks)
        if entry.get("sampling_order"):
            order_info = entry["sampling_order"]
            order_path = path.parent / order_info["file"]
            if order_path.parent != path.parent:
                raise ValueError(
                    "Sampling order must be local to the frozen task directory"
                )
            raw = order_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != order_info["sha256"]:
                raise ValueError("Frozen GSM8K sampling order changed")
            indices = json.loads(raw)["indices"]
            if len(indices) != order_info["entries"] or any(
                type(i) is not int or not 0 <= i < source_count for i in indices
            ):
                raise ValueError("Invalid GSM8K sampling order")
            tasks = [tasks[index] for index in indices]
        return tasks, {
            **manifest,
            **entry,
            "source_count": source_count,
            "training_allowed": entry["split"] == "train",
        }

    @staticmethod
    def prepare(task: GSM8KTask) -> TaskInput:
        return TaskInput(
            task.task_id, task.question + PROMPT_SUFFIX, EnvironmentSpec("python")
        )

    @staticmethod
    def grade(task: GSM8KTask, result: TaskResult, output: Path) -> GradeResult:
        if not result.completed:
            raise RuntimeError("Incomplete execution is not a GSM8K quality score")
        from areal.workflow.openai.math_agent import math_reward_fn

        # Keep full prediction/gold and the original verify(prediction, gold)
        # order. The library's own parsing/verification defaults are unchanged;
        # exceptions escaping the original function propagate to the client.
        reward = math_reward_fn(completions=result.answer, answer=task.answer)
        return GradeResult(reward, "correct" if reward == 1.0 else "incorrect")

    @staticmethod
    def result_metadata(task: GSM8KTask) -> dict[str, Any]:
        return {
            "source_split": task.split,
            "source_index": task.source_index,
            "gold_answer": task.answer,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Existing local GSM8K main dataset"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New private frozen task directory"
    )
    args = parser.parse_args()
    from datasets import load_dataset

    splits = {
        split: load_dataset(str(args.dataset), "main", split=split)
        for split in ("train", "test")
    }
    source = {
        "path": str(args.dataset.resolve()),
        "fingerprints": {split: rows._fingerprint for split, rows in splits.items()},
        "length_filter": None,
    }
    freeze_splits(splits, args.output, source=source)


if __name__ == "__main__":
    main()
