# SPDX-License-Identifier: Apache-2.0

"""Resource-independent, seeded synthetic episode DAGs; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median

VERSION = "synthetic-workload-v2"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _rng(seed: int, *identity: object) -> random.Random:
    return random.Random(int(digest((VERSION, seed, identity)), 16))


@dataclass(frozen=True)
class LogNormal:
    """Clipped lognormal; median/sigma describe the distribution before clipping."""

    median: float
    sigma: float
    minimum: float
    maximum: float

    def __post_init__(self):
        if (
            not all(math.isfinite(v) for v in asdict(self).values())
            or not 0 < self.minimum <= self.median <= self.maximum
            or self.sigma < 0
        ):
            raise ValueError("Require positive finite lognormal bounds and sigma >= 0")

    def sample(self, rng: random.Random, *, integer: bool = False) -> float | int:
        value = min(
            self.maximum,
            max(self.minimum, rng.lognormvariate(math.log(self.median), self.sigma)),
        )
        return max(1, round(value)) if integer else value


@dataclass(frozen=True)
class TaskFamily:
    name: str
    prompt: LogNormal
    output: LogNormal
    tool_output: LogNormal
    tool_work: LogNormal
    mean_turns: float
    max_turns: int
    tool_type: str
    tool_reference: str
    message_overhead_tokens: int = 16

    def __post_init__(self):
        if (
            not self.name
            or not self.tool_type
            or not self.tool_reference
            or not math.isfinite(self.mean_turns)
            or self.mean_turns < 1
            or type(self.max_turns) is not int
            or self.max_turns < 1
            or type(self.message_overhead_tokens) is not int
            or self.message_overhead_tokens < 0
            or any(
                d.minimum < 1
                or int(d.minimum) != d.minimum
                or int(d.maximum) != d.maximum
                for d in (self.prompt, self.output, self.tool_output)
            )
        ):
            raise ValueError("Invalid task family, token bounds or tool reference")

    @classmethod
    def decode(cls, value: dict) -> TaskFamily:
        return cls(
            **{
                **value,
                **{
                    k: LogNormal(**value[k])
                    for k in ("prompt", "output", "tool_output", "tool_work")
                },
            }
        )


@dataclass(frozen=True)
class ArrivalPhase:
    until_s: float
    rate_per_s: float


@dataclass(frozen=True)
class CompositionPhase:
    until_s: float
    weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class WorkloadSpec:
    seed: int
    duration_s: float
    arrivals: tuple[ArrivalPhase, ...]
    composition: tuple[CompositionPhase, ...]
    families: tuple[TaskFamily, ...]

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or self.seed < 0
            or not math.isfinite(self.duration_s)
            or self.duration_s <= 0
        ):
            raise ValueError("Require a nonnegative integer seed and positive duration")
        names = {f.name for f in self.families}
        if not names or len(names) != len(self.families):
            raise ValueError("Task families must be nonempty and unique")
        for phases in (self.arrivals, self.composition):
            ends = [p.until_s for p in phases]
            if (
                not ends
                or ends[-1] != self.duration_s
                or any(
                    not math.isfinite(end) or end <= start
                    for start, end in zip([0] + ends[:-1], ends, strict=True)
                )
            ):
                raise ValueError("Each schedule must partition [0, duration_s)")
        if len(self.composition) != 1:
            raise ValueError("Task composition must be fixed over the arrival window")
        if any(
            not math.isfinite(p.rate_per_s) or p.rate_per_s < 0 for p in self.arrivals
        ):
            raise ValueError("Poisson rates must be finite and nonnegative")
        for phase in self.composition:
            weights = dict(phase.weights)
            if (
                len(weights) != len(phase.weights)
                or not weights.keys() <= names
                or not weights
                or any(not math.isfinite(w) or w < 0 for w in weights.values())
                or sum(weights.values()) <= 0
            ):
                raise ValueError(
                    "Composition requires known families and positive total weight"
                )

    @classmethod
    def decode(cls, value: dict) -> WorkloadSpec:
        value = dict(value)
        if value.pop("schema_version") != VERSION:
            raise ValueError("Unknown synthetic workload schema")
        return cls(
            **{
                **value,
                "arrivals": tuple(ArrivalPhase(**p) for p in value["arrivals"]),
                "composition": tuple(
                    CompositionPhase(p["until_s"], tuple(sorted(p["weights"].items())))
                    for p in value["composition"]
                ),
                "families": tuple(TaskFamily.decode(f) for f in value["families"]),
            }
        )

    def encode(self) -> dict:
        value = asdict(self)
        value["schema_version"] = VERSION
        for p in value["composition"]:
            p["weights"] = dict(p["weights"])
        return value


def arrival_times(spec: WorkloadSpec) -> tuple[float, ...]:
    """Invert integrated intensity for a shared unit-rate Poisson stream.

    Equal integrated intensity and seed preserve count/episode identities when
    only the temporal rate shape changes. Zero-rate intervals consume no hazard.
    """
    rng = _rng(spec.seed, "arrivals")
    times, start = [], 0.0
    remaining = rng.expovariate(1.0)
    for phase in spec.arrivals:
        intensity = (phase.until_s - start) * phase.rate_per_s
        while remaining < intensity:
            times.append(start + remaining / phase.rate_per_s)
            remaining += rng.expovariate(1.0)
        remaining -= intensity
        start = phase.until_s
    return tuple(times)


def generate(spec: WorkloadSpec) -> tuple[dict, ...]:
    families = {f.name: f for f in spec.families}
    names, weights = zip(*spec.composition[0].weights, strict=True)
    episodes = []
    for index, at in enumerate(arrival_times(spec)):
        episode_id = f"episode-{index:08d}"
        name = _rng(spec.seed, episode_id, "family").choices(names, weights, k=1)[0]
        family = families[name]
        rng = _rng(spec.seed, episode_id, "turn_count")
        turns = 1
        while turns < family.max_turns and rng.random() >= 1 / family.mean_turns:
            turns += 1
        prompt = family.prompt.sample(
            _rng(spec.seed, episode_id, "initial_prompt"), integer=True
        )
        nodes, predecessors = [], []
        for turn in range(turns):
            output = family.output.sample(
                _rng(spec.seed, episode_id, turn, "output"), integer=True
            )
            name = f"request-{turn}"
            nodes.append(
                {
                    "id": name,
                    "kind": "inference",
                    "predecessors": predecessors,
                    "input_tokens": prompt,
                    "output_tokens": output,
                }
            )
            predecessors = [name]
            prompt += output + family.message_overhead_tokens
            if turn + 1 < turns:
                reply = family.tool_output.sample(
                    _rng(spec.seed, episode_id, turn, "tool_output"), integer=True
                )
                tool = f"tool-{turn}"
                nodes.append(
                    {
                        "id": tool,
                        "kind": "tool",
                        "predecessors": predecessors,
                        "tool_type": family.tool_type,
                        "output_tokens": reply,
                        "work_units": family.tool_work.sample(
                            _rng(spec.seed, episode_id, turn, "tool_work")
                        ),
                        "work_reference": family.tool_reference,
                    }
                )
                prompt += reply
                predecessors = [tool]
        episodes.append(
            {
                "episode_id": episode_id,
                "task_id": index,
                "available_at": at,
                "family": family.name,
                "nodes": nodes,
            }
        )
    return tuple(episodes)


def write_workload(
    output: Path, spec: WorkloadSpec, episodes: tuple[dict, ...]
) -> dict:
    """Freeze pure business inputs; no batching, deployment, model or timestamps of execution."""
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("spec.json", "workload.jsonl", "manifest.json")
    ):
        raise FileExistsError(
            "Workload files already exist; choose a new output directory"
        )
    data = "".join(canonical(e) + "\n" for e in episodes).encode()
    starts = [0.0] + [p.until_s for p in spec.arrivals[:-1]]
    mix_starts = [0.0] + [p.until_s for p in spec.composition[:-1]]
    requests = [n for e in episodes for n in e["nodes"] if n["kind"] == "inference"]
    tools = [n for e in episodes for n in e["nodes"] if n["kind"] == "tool"]

    def summary(values):
        values = sorted(values)
        return {
            "count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1]
            if values
            else None,
        }

    manifest = {
        "schema_version": VERSION,
        "seed": spec.seed,
        "duration_s": spec.duration_s,
        "time_unit": "seconds since external workload start",
        "spec_sha256": digest(spec.encode()),
        "trace_sha256": hashlib.sha256(data).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "episodes": len(episodes),
        "inference_requests": len(requests),
        "tool_calls": len(tools),
        "sequence_semantics": "R inference requests alternate with R-1 tool calls; final inference ends episode",
        "composition_semantics": "fixed probabilities, independent of external arrival times",
        "families": dict(Counter(e["family"] for e in episodes)),
        "arrival_phases": [
            {
                "from_s": start,
                "until_s": p.until_s,
                "expected_episodes": (p.until_s - start) * p.rate_per_s,
                "realized_episodes": sum(
                    start <= e["available_at"] < p.until_s for e in episodes
                ),
            }
            for start, p in zip(starts, spec.arrivals, strict=True)
        ],
        "composition_phases": [
            {
                "from_s": start,
                "until_s": p.until_s,
                "weights": dict(p.weights),
                "realized_families": dict(
                    Counter(
                        e["family"]
                        for e in episodes
                        if start <= e["available_at"] < p.until_s
                    )
                ),
            }
            for start, p in zip(mix_starts, spec.composition, strict=True)
        ],
        "statistics": {
            "turns": summary(
                sum(n["kind"] == "inference" for n in e["nodes"]) for e in episodes
            ),
            "input_tokens": summary(n["input_tokens"] for n in requests),
            "output_tokens": summary(n["output_tokens"] for n in requests),
            "tool_output_tokens": summary(n["output_tokens"] for n in tools),
            "tool_work_units": summary(n["work_units"] for n in tools),
        },
        "token_unit": "synthetic tokens; no text/token IDs or reward prediction",
        "tool_units": "work_reference identifies baseline work; experiment must supply seconds per unit",
        "arrival_semantics": "external submission, before admission or queueing; no batch padding/truncation",
    }
    (output / "spec.json").write_text(canonical(spec.encode()) + "\n")
    (output / "workload.jsonl").write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_workload(output: Path) -> tuple[dict, tuple[dict, ...]]:
    manifest = json.loads((output / "manifest.json").read_text())
    data = (output / "workload.jsonl").read_bytes()
    if (
        manifest["schema_version"] not in ("synthetic-workload-v1", VERSION)
        or hashlib.sha256(data).hexdigest() != manifest["trace_sha256"]
    ):
        raise ValueError("Workload version or trace digest differs from manifest")
    spec = json.loads((output / "spec.json").read_text())
    if (
        spec.get("schema_version") != manifest["schema_version"]
        or digest(spec) != manifest["spec_sha256"]
    ):
        raise ValueError("Workload generation spec differs from manifest")
    episodes = tuple(json.loads(line) for line in data.splitlines())
    if len(episodes) != manifest["episodes"]:
        raise ValueError("Workload episode count differs from manifest")
    return manifest, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--arrivals", type=Path, help="Override arrival phases with a JSON array"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = WorkloadSpec.decode(json.loads(args.spec.read_text()))
    if args.arrivals is not None:
        spec = replace(
            spec,
            arrivals=tuple(
                ArrivalPhase(**p) for p in json.loads(args.arrivals.read_text())
            ),
        )
    manifest = write_workload(args.output, spec, generate(spec))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
