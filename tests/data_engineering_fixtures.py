# SPDX-License-Identifier: Apache-2.0

"""Small external recipes exercising the public data execution contract."""

from dataclasses import replace

from areal.v2.data_engineering import (
    Attempt,
    CalibrationBatch,
    DataSample,
    InsufficientDataError,
    PublicTask,
    Selection,
)


def sample(task_id, value, reference):
    return DataSample(PublicTask(task_id, {"value": value}), reference)


class FixtureRecipe:
    def __init__(self, mode="normal"):
        self.mode = mode

    def generate(self, context):
        if self.mode == "dirty_reference":
            return [
                sample("dirty", "bad", float("nan")),
                sample("a", "1", "1"),
                sample("b", "2", "2"),
            ]
        if self.mode == "shortage":
            return [
                sample(f"batch-{context.batch_index}", str(context.batch_index), "0")
            ]
        if self.mode == "conflict":
            return [sample("a", "1", "1"), sample("b", "1", "2")]
        if self.mode == "id_conflict":
            return [sample("a", "1", "1"), sample("a", "2", "2")]
        if self.mode == "alias_conflict":
            return [sample("a", "1", "1"), sample("b", "1", "1"), sample("b", "2", "2")]
        return [
            sample("a", " 1 ", "1"),
            sample("a", " 1 ", "1"),
            sample("same-input", "1", "1"),
            sample("invalid", "", "0"),
            sample("excluded", "2", "2"),
            sample("b", "3", "3"),
            sample("c", "4", "4"),
        ]

    def clean(self, item):
        if isinstance(item.reference, float):
            return None
        value = item.task.input["value"].strip()
        return (
            replace(item, task=replace(item.task, input={"value": value}))
            if value
            else None
        )

    def score(self, item, response):
        return float("nan") if self.mode == "nan" else float(response == item.reference)

    def select(self, samples, scores, context):
        if self.mode == "exception":
            raise RuntimeError("Broken selector")
        if self.mode == "change_source":
            context.inputs["source"].write_text("changed")
        available = [
            item
            for item in samples
            if scores is None or None not in scores[item.task.task_id]
        ]
        if len(available) < context.target_count:
            raise InsufficientDataError("Not enough fixture tasks")
        ids = tuple(item.task.task_id for item in available[: context.target_count])
        return Selection(
            (ids[0],) * context.target_count if self.mode == "bad_selection" else ids
        )

    def training_row(self, item):
        if self.mode == "mutating_projection":
            item.task.input["value"] = "formatted training input"
        if self.mode == "bad_row":
            return {"bad": object()}
        return {
            "task_id": item.task.task_id,
            "prompt": item.task.input["value"],
            "answer": item.reference,
        }


def create_recipe(**params):
    return FixtureRecipe(**params)


class FixtureCalibrator:
    def __init__(self, mode="normal"):
        self.mode = mode

    async def __call__(self, tasks, context):
        attempts = []
        for task in tasks:
            assert not hasattr(task, "reference")
            assert set(task.input) == {"value"}
            value = task.input["value"]
            if self.mode == "mutate":
                task.input["value"] = "mutated backend input"
            for repeat in range(context.repeats):
                error = (
                    "backend_timeout"
                    if self.mode == "failure" and task.task_id == "a"
                    else None
                )
                attempts.append(Attempt(task.task_id, repeat, value, error))
        if self.mode == "missing":
            attempts.pop()
        elif self.mode == "duplicate":
            attempts.append(attempts[0])
        elif self.mode == "unknown":
            attempts[0] = Attempt("unknown", 0, "0")
        elif self.mode == "bad_repeat":
            attempts[0] = Attempt(attempts[0].task_id, True, "0")
        elif self.mode == "cancel":
            import asyncio

            raise asyncio.CancelledError()
        assert not hasattr(context, "model_ref")
        assert not hasattr(context, "harness_ref")
        return CalibrationBatch(
            tuple(attempts),
            [] if self.mode == "bad_provenance" else {"method": "fixture-rules-v1"},
        )


def create_calibrator(**params):
    return FixtureCalibrator(**params)
