# SPDX-License-Identifier: Apache-2.0

"""Exact-answer mathematical task recipe for TIR.

Domain formulas and selection policy stay here; execution and storage live in
areal.v2.data_engineering. This produces tasks, not teacher trajectories.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from fractions import Fraction
from math import comb, lcm
from pathlib import Path
from typing import Any

from areal.v2.data_engineering.selection import select_ranked_candidates
from areal.v2.data_engineering.types import (
    DataSample,
    GenerationShard,
    InsufficientDataError,
    PublicTask,
    RankedCandidate,
    RecipeContext,
    Selection,
    SelectionPolicy,
    positive_int,
)


class Family(StrEnum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    COUNTING = "counting"
    PROBABILITY = "probability"
    GEOMETRY = "geometry"
    RECURRENCE = "recurrence"
    OPTIMIZATION = "optimization"
    RATES = "rates"


def _diverse_problem(
    family: Family, variant: int, rng: random.Random
) -> tuple[str, str]:
    """Exact-answer constructions; each variant changes the mathematical structure."""
    a, b, c = (rng.randint(3, 100) for _ in range(3))
    if family is Family.ALGEBRA:
        if variant == 0:
            return (
                f"Real numbers x and y satisfy x+y={a + b} and xy={a * b}. Find x^3+y^3.",
                str(a**3 + b**3),
            )
        return (
            f"A monic quadratic polynomial P satisfies P({a})={a * a + b * a + c} and "
            f"P({a + 2})={(a + 2) ** 2 + b * (a + 2) + c}. Find P({-b}).",
            str(c),
        )
    if family is Family.NUMBER_THEORY:
        if variant == 0:
            modulus = rng.choice((7, 11, 13, 17, 19, 23))
            exponent = rng.randint(23, 120)
            return (
                f"Find the least nonnegative remainder of {a}^{exponent}+{b}^{exponent + 1} "
                f"when divided by {modulus}.",
                str(
                    (pow(a, exponent, modulus) + pow(b, exponent + 1, modulus))
                    % modulus
                ),
            )
        p, q = rng.sample((3, 5, 7, 11, 13, 17), 2)
        target = rng.randint(1, p * q - 1)
        lower = rng.randint(3, 30) * p * q
        return (
            f"Find the smallest integer n greater than {lower} such that n leaves remainder "
            f"{target % p} modulo {p} and remainder {target % q} modulo {q}.",
            str(lower + target),
        )
    if family is Family.COUNTING:
        if variant == 0:
            n, k = rng.randint(25, 200), rng.randint(3, 12)
            return (
                f"How many subsets of size {k} of {{1,2,...,{n}}} "
                "contain no two consecutive integers?",
                str(comb(n - k + 1, k)),
            )
        total, bound = rng.randint(15, 200), rng.randint(3, 14)
        return (
            f"How many ordered triples (x,y,z) of nonnegative integers satisfy "
            f"x+y+z={total} and x<={bound}?",
            str(comb(total + 2, 2) - comb(total - bound + 1, 2)),
        )
    if family is Family.PROBABILITY:
        red, blue = rng.randint(5, 60), rng.randint(5, 60)
        if variant == 0:
            k = rng.randint(3, 5)
            return (
                f"A bag contains {red} red and {blue} blue balls. Draw {k} balls uniformly "
                "without replacement. What is the probability of exactly two red balls? "
                "Give a reduced fraction.",
                str(Fraction(comb(red, 2) * comb(blue, k - 2), comb(red + blue, k))),
            )
        return (
            f"A bag contains {red} red and {blue} blue balls. Two balls are drawn uniformly "
            "without replacement. Given that at least one is red, what is the probability "
            "that both are red? Give a reduced fraction.",
            str(Fraction(comb(red, 2), comb(red + blue, 2) - comb(blue, 2))),
        )
    if family is Family.GEOMETRY:
        if variant == 0:
            scale = rng.randint(2, 35)
            x, y = rng.randint(-30, 30), rng.randint(-30, 30)
            return (
                f"A triangle has vertices ({x},{y}), ({x + 3 * scale},{y}), and "
                f"({x},{y + 4 * scale}). Find the radius of its inscribed circle.",
                str(scale),
            )
        width, height = rng.randint(5, 30), rng.randint(5, 30)
        cut = rng.randint(1, min(width, height) // 2)
        return (
            f"A rectangle has width {width} and height {height}. From each corner a "
            f"right isosceles triangle with both legs {cut} along the sides is removed. "
            "Find the area remaining.",
            str(width * height - 2 * cut * cut),
        )
    if family is Family.RECURRENCE:
        if variant == 0:
            n = rng.randint(5, 12)
            ratio = rng.randint(2, 4)
            answer = a
            for _ in range(n - 1):
                answer = ratio * answer + b
            return (
                f"A sequence satisfies u_1={a} and u_(n+1)={ratio}u_n+{b} for n>=1. Find u_{n}.",
                str(answer),
            )
        n = rng.randint(8, 24)
        return (
            f"A sequence satisfies u_1={a}, u_2={a + b + c}, and "
            f"u_(n+2)-2u_(n+1)+u_n={2 * c}. Find u_{n}.",
            str(a + (n - 1) * b + c * (n - 1) ** 2),
        )
    if family is Family.OPTIMIZATION:
        if variant == 0:
            total = rng.randint(20, 2000)
            return (
                f"Positive integers x,y satisfy x+y={total}. Find the maximum value of xy+x.",
                str(max(x * (total - x) + x for x in range(1, total))),
            )
        budget = rng.randint(100, 400)
        cost1, cost2 = rng.sample(range(7, 25), 2)
        reward1, reward2 = rng.randint(10, 30), rng.randint(10, 30)
        best = max(
            reward1 * x + reward2 * y
            for x in range(budget // cost1 + 1)
            for y in range((budget - cost1 * x) // cost2 + 1)
        )
        return (
            f"Nonnegative integers x,y satisfy {cost1}x+{cost2}y<={budget}. "
            f"Find the maximum value of {reward1}x+{reward2}y.",
            str(best),
        )
    if family is Family.RATES:
        if variant == 0:
            t1, t2 = rng.randint(15, 40), rng.randint(15, 40)
            elapsed = rng.randint(2, 6)
            remaining = 1 - Fraction(elapsed, t1) - Fraction(elapsed, t2)
            return (
                f"Pipe A fills a tank in {t1} hours and pipe B in {t2} hours. Both "
                f"start filling an empty tank. After {elapsed} hours B is closed. "
                "How many total hours from the start are needed to fill the tank? "
                "Give an exact answer.",
                str(elapsed + remaining * t1),
            )
        speed, current = rng.randint(15, 40), rng.randint(2, 8)
        distance = lcm(speed - current, speed + current) * rng.randint(1, 5)
        return (
            f"A boat travels at {speed} km/h in still water. It travels {distance} km "
            f"downstream then {distance} km upstream in a current of {current} km/h. "
            "Find its average speed for the whole trip in km/h as a reduced fraction.",
            str(
                Fraction(
                    2 * distance,
                    Fraction(distance, speed + current)
                    + Fraction(distance, speed - current),
                )
            ),
        )
    raise ValueError(f"Unsupported family {family}")


def _answer(response: str) -> Fraction | None:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    end = start
    while end < len(response) and depth:
        if response[end] == "{":
            depth += 1
        elif response[end] == "}":
            depth -= 1
        end += 1
    if depth:
        return None
    text = response[start : end - 1].strip()
    fraction = re.fullmatch(
        r"\\(?:d?frac)\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", text
    )
    if fraction:
        text = "/".join(fraction.groups())
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


class TirDataRecipe:
    def __init__(
        self,
        candidate_count: int = 1024,
        minimum_one_pass: int | None = None,
        maximum_per_recipe: int | None = None,
    ):
        positive_int(candidate_count, "candidate_count", minimum=16)
        if candidate_count % 16:
            raise ValueError("candidate_count must be divisible by 16")
        if minimum_one_pass is not None:
            positive_int(minimum_one_pass, "minimum_one_pass", minimum=0)
        if maximum_per_recipe is not None:
            positive_int(maximum_per_recipe, "maximum_per_recipe")
        self.candidate_count = candidate_count
        self.minimum_one_pass = minimum_one_pass
        self.maximum_per_recipe = maximum_per_recipe

    def source_files(self) -> tuple[Path, ...]:
        return (Path(__file__),)

    def generation_shards(
        self, context: RecipeContext, shard_size: int
    ) -> Iterable[GenerationShard]:
        # Independent streams preserve uniqueness within each mathematical recipe.
        count = self.candidate_count // 16
        for group in range(16):
            yield GenerationShard(group * count, (group + 1) * count)

    def generate_shard(
        self, context: RecipeContext, shard: GenerationShard
    ) -> Iterable[DataSample]:
        count = self.candidate_count // 16
        group = shard.start // count
        if (
            group >= 16
            or shard.start != group * count
            or shard.stop != (group + 1) * count
        ):
            raise ValueError(
                "TIR generation needs one complete recipe stream per shard"
            )
        family, variant = tuple(Family)[group % 8], group // 8
        rng = random.Random(f"tir:{context.seed}:{context.batch_index}:{group}")
        seen = set()
        for _ in range(count):
            for _ in range(1000):
                question, answer = _diverse_problem(family, variant, rng)
                task_id = hashlib.sha256(
                    " ".join(question.casefold().split()).encode()
                ).hexdigest()
                if task_id not in seen:
                    break
            else:
                raise InsufficientDataError(
                    "TIR recipe exhausted its distinct parameters"
                )
            seen.add(task_id)
            yield DataSample(
                PublicTask(
                    task_id,
                    {"messages": [{"role": "user", "content": question}]},
                    {
                        "family": family.value,
                        "recipe": f"{family.value}_{variant}",
                        "structure_level": 2 + variant,
                    },
                ),
                answer,
            )

    def generate(self, context: RecipeContext) -> list[DataSample]:
        return [
            sample
            for shard in self.generation_shards(context, self.candidate_count)
            for sample in self.generate_shard(context, shard)
        ]

    def clean(self, sample: DataSample) -> DataSample | None:
        messages = sample.task.input.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
        ):
            return None
        question = messages[0].get("content")
        if (
            messages[0].get("role") != "user"
            or not isinstance(question, str)
            or not question.strip()
        ):
            return None
        if sample.task.metadata.get("family") not in {
            family.value for family in Family
        }:
            return None
        try:
            Fraction(sample.reference)
        except (ValueError, TypeError, ZeroDivisionError):
            return None
        expected = hashlib.sha256(
            " ".join(question.casefold().split()).encode()
        ).hexdigest()
        return sample if sample.task.task_id == expected else None

    def score(self, sample: DataSample, response: Any) -> float:
        if not isinstance(response, str):
            raise ValueError("TIR calibration responses must be strings")
        return float(_answer(response) == Fraction(sample.reference))

    def _required_one_pass(self, context: RecipeContext, calibrated: bool) -> int:
        count = context.target_count
        if count < 8 or count % 8:
            raise ValueError("TIR target_count must be divisible by eight")
        required = (
            0
            if not calibrated
            else (
                count // 4 if self.minimum_one_pass is None else self.minimum_one_pass
            )
        )
        if required > count:
            raise ValueError("minimum_one_pass exceeds target_count")
        return required

    def selection_policy(
        self, context: RecipeContext, calibrated: bool
    ) -> SelectionPolicy:
        self._required_one_pass(context, calibrated)
        minimum = max(1, context.target_count // (2 * len(Family)))
        return SelectionPolicy(
            {family.value: minimum for family in Family},
            {}
            if self.maximum_per_recipe is None
            else {
                f"{family.value}_{variant}": self.maximum_per_recipe
                for family in Family
                for variant in range(2)
            },
        )

    def rank(
        self,
        sample: DataSample,
        scores: tuple[float | None, ...] | None,
        context: RecipeContext,
    ) -> RankedCandidate | None:
        passed = None
        if scores is not None:
            if len(scores) != 2 or any(
                value not in {None, 0.0, 1.0} for value in scores
            ):
                raise ValueError("TIR calibration requires two binary-scored attempts")
            if None in scores or sum(scores) >= 2:
                return None
            passed = int(sum(scores))
        return RankedCandidate(
            sample.task.task_id,
            0 if passed == 1 else 1,
            hashlib.sha256(
                f"{context.seed}:{sample.task.task_id}".encode()
            ).hexdigest(),
            sample.task.metadata["family"],
            sample.task.metadata["recipe"],
            {"passed": passed},
        )

    def finalize_selection(
        self,
        selected: Sequence[RankedCandidate],
        eligible_count: int,
        context: RecipeContext,
    ) -> Selection:
        one_pass = sum(item.metadata["passed"] == 1 for item in selected)
        calibrated = any(item.metadata["passed"] is not None for item in selected)
        if one_pass < self._required_one_pass(context, calibrated):
            raise InsufficientDataError(
                "Too few borderline cases; revise generator difficulty"
            )
        ordered = sorted(
            selected,
            key=lambda item: hashlib.sha256(
                f"train:{context.seed}:{item.task_id}".encode()
            ).hexdigest(),
        )
        return Selection(
            tuple(item.task_id for item in ordered),
            {
                "eligible": eligible_count,
                "one_pass": one_pass if calibrated else None,
                "zero_pass": len(selected) - one_pass if calibrated else None,
                "families": dict(Counter(item.group for item in selected)),
                "recipes": dict(Counter(item.bucket for item in selected)),
            },
        )

    def select(
        self,
        samples: Sequence[DataSample],
        scores: Mapping[str, tuple[float | None, ...]] | None,
        context: RecipeContext,
    ) -> Selection:
        policy = self.selection_policy(context, scores is not None)
        ranked = [
            item
            for sample in samples
            if (
                item := self.rank(
                    sample,
                    scores[sample.task.task_id] if scores is not None else None,
                    context,
                )
            )
            is not None
        ]
        ranked.sort(key=lambda item: (item.priority, item.tie_breaker, item.task_id))
        selected = select_ranked_candidates(
            policy,
            context.target_count,
            lambda group: (
                item for item in ranked if group is None or item.group == group
            ),
        )
        return self.finalize_selection(selected, len(ranked), context)

    def training_row(self, sample: DataSample) -> dict[str, Any]:
        return {
            "task_id": sample.task.task_id,
            "messages": sample.task.input["messages"],
            "answer": f"\\boxed{{{sample.reference}}}",
        }


def create_recipe(**params: Any) -> TirDataRecipe:
    return TirDataRecipe(**params)
