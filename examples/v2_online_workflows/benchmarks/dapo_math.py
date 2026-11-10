# SPDX-License-Identifier: Apache-2.0

"""Pinned DAPO tasks and final-answer grading, kept outside the agent context."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.types import EnvironmentSpec, TaskInput, TaskResult

DATASET = "open-r1/DAPO-Math-17k-Processed"
INSTRUCTION = (
    "Solve the following math problem. You may use math_python to compute, "
    "enumerate or check intermediate results when helpful. Each Python call uses "
    "a fresh interpreter; files persist within this task. Explain your solution "
    "and put your final answer in \\boxed{...}.\n\n"
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def question_id(question: str) -> str:
    return "dapo-" + digest_bytes(question.encode())


@dataclass(frozen=True)
class MathTask:
    task_id: str
    question: str
    gold_answer: str

    def prompt(self) -> str:
        return INSTRUCTION + self.question


@dataclass(frozen=True)
class MathScore:
    reward: float
    reason: str
    extracted_answer: str | None


def tasks_from_rows(rows: Any) -> list[MathTask]:
    """Keep exact upstream questions/labels; validate stable task identities."""
    tasks: dict[str, MathTask] = {}
    for row in rows:
        question = row["prompt"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Expected the processed DAPO prompt string")
        answer = str(
            row["solution"]
            if "solution" in row
            else row["reward_model"]["ground_truth"]
        ).strip()
        if not answer:
            raise ValueError("Empty DAPO gold answer")
        task = MathTask(question_id(question), question, answer)
        previous = tasks.get(task.task_id)
        if previous is not None and previous.gold_answer != task.gold_answer:
            raise ValueError(
                f"Inconsistent labels for identical raw task {task.task_id}"
            )
        tasks.setdefault(task.task_id, task)
    return [tasks[identity] for identity in sorted(tasks)]


def extract_final_answer(text: str) -> str | None:
    """Take the last explicit final answer, never an arbitrary reasoning number."""
    candidates: list[tuple[int, str]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                candidates.append((match.start(), text[start:index].strip()))
                break
    for match in re.finditer(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", text):
        candidates.append((match.start(), match.group(1).strip()))
    if not candidates:
        stripped = text.strip()
        return stripped if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped) else None
    answer = max(candidates, key=lambda item: item[0])[1]
    return answer if answer and len(answer) <= 256 else None


def score_answer(text: str, gold: str, *, completed: bool = True) -> MathScore:
    """Grade only the final answer; verifier exceptions remain infrastructure errors."""
    if not completed:
        return MathScore(0.0, "iteration_or_generation_limit", None)
    answer = extract_final_answer(text)
    if answer is None:
        return MathScore(0.0, "missing_final_answer", None)
    # DAPO golds are numerical. Exact rational comparison avoids approximate
    # float equality and keeps the common path independent of a CAS timeout.
    try:
        correct = Fraction(answer.strip("$ ")) == Fraction(gold.strip("$ "))
    except (ValueError, ZeroDivisionError):
        from math_verify import parse, verify

        gold_parsed = parse(gold, parsing_timeout=None)
        if not gold_parsed:
            raise ValueError("The gold answer cannot be parsed")
        prediction = parse("\\boxed{" + answer + "}", parsing_timeout=None)
        if not prediction:
            return MathScore(0.0, "unparseable_final_answer", answer)
        correct = verify(gold_parsed, prediction, timeout_seconds=None)
    return MathScore(float(correct), "correct" if correct else "incorrect", answer)


def write_splits(
    tasks: list[MathTask],
    output: Path,
    *,
    source: dict[str, Any],
    seed: int,
    pilot: int,
    train: int,
    evaluation: int,
) -> dict[str, Any]:
    counts = {"pilot": pilot, "train": train, "eval": evaluation}
    if min(counts.values()) < 1 or sum(counts.values()) > len(tasks):
        raise ValueError("Positive split sizes must fit the unique task pool")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Task pool must be deduplicated before splitting")
    groups: dict[str, list[MathTask]] = {}
    for task in sorted(tasks, key=lambda task: task.task_id):
        groups.setdefault(" ".join(task.question.split()), []).append(task)
    remaining = list(groups.values())
    random.Random(seed).shuffle(remaining)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1,
        "source": source,
        "seed": seed,
        "unique_tasks": len(tasks),
        "split_grouping": "whitespace_normalized_question; upstream records preserved",
        "format_variant_groups": sum(len(group) > 1 for group in remaining),
        "splits": {},
    }
    for split, count in counts.items():
        selected: list[MathTask] = []
        unselected = []
        for group in remaining:
            if len(group) <= count - len(selected):
                selected.extend(group)
            else:
                unselected.append(group)
        if len(selected) != count:
            raise ValueError(
                "Split sizes cannot be filled without separating question variants"
            )
        remaining = unselected
        payload = "".join(
            json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in selected
        )
        path = output / f"{split}.jsonl"
        path.write_text(payload, encoding="utf-8")
        manifest["splits"][path.name] = {
            "count": count,
            "sha256": digest_bytes(payload.encode()),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_tasks(path: Path) -> tuple[list[MathTask], dict[str, Any]]:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    payload = path.read_bytes()
    split = manifest["splits"][path.name]
    if manifest["format_version"] != 1 or digest_bytes(payload) != split["sha256"]:
        raise ValueError("Task file differs from its frozen manifest")
    tasks = [
        MathTask(**json.loads(line))
        for line in payload.decode().splitlines()
        if line.strip()
    ]
    if len(tasks) != split["count"] or len({task.task_id for task in tasks}) != len(
        tasks
    ):
        raise ValueError("Task count or identity differs from the manifest")
    for task in tasks:
        if task.task_id != question_id(task.question) or not task.gold_answer.strip():
            raise ValueError("Invalid task identity or gold label")
    return tasks, manifest


class DapoBenchmark:
    """The frozen math task/score contract, with no agent-framework dependency."""

    name = "dapo_math"
    request_prefix = "math"
    load_tasks = staticmethod(load_tasks)

    @staticmethod
    def restore_task(value: dict[str, Any]) -> MathTask:
        return MathTask(**value)

    def prepare(self, task: MathTask) -> TaskInput:
        return TaskInput(task.task_id, task.prompt(), EnvironmentSpec("python"))

    def grade(self, task: MathTask, result: TaskResult, output: Path) -> MathScore:
        return score_answer(result.answer, task.gold_answer, completed=result.completed)

    def result_metadata(self, task: MathTask) -> dict[str, Any]:
        return {"gold_answer": task.gold_answer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", default="main", help="Resolved to a commit before download"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pilot", type=int, default=200)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--eval", type=int, default=200)
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Cannot resolve an immutable dataset revision")
    dataset = load_dataset(DATASET, "en", split="train", revision=revision)
    tasks = tasks_from_rows(dataset)
    manifest = write_splits(
        tasks,
        args.output,
        source={
            "dataset": DATASET,
            "revision": revision,
            "config": "en",
            "split": "train",
            "rows": len(dataset),
        },
        seed=args.seed,
        pilot=args.pilot,
        train=args.train,
        evaluation=args.eval,
    )
    sys.stdout.write(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
