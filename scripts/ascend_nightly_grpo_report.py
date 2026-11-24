"""Generate the Ascend GRPO job summary from the current and optional baseline main.log."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

GRPO_METRICS = {
    "rollout/reward": "Rollout reward",
    "eval-rollout/reward": "Validation reward",
    "rollout/spec_decode/acceptance_rate": "Acceptance rate",
    "rollout/spec_decode/mean_accepted_len": "Mean accepted length",
}
SFT_METRICS = {
    "sft/loss/avg": "Training loss",
    "sft-eval/loss/avg": "Validation loss",
    "sft/ppl/avg": "Training perplexity",
    "sft-eval/ppl/avg": "Validation perplexity",
}
METRICS = GRPO_METRICS | SFT_METRICS
MAIN_TIMERS = (
    "rollout",
    "train_step",
    "update_weights",
    "recompute_logp",
    "compute_advantage",
    "eval",
)
TIME = re.compile(r"\d{8}-\d\d:\d\d:\d\d\.\d{3}")
STEP = re.compile(r"Epoch (\d+)/(\d+) Step (\d+)/(\d+) Train step (\d+)/(\d+) done\.")


@dataclass
class Step:
    epoch: int
    number: int
    epoch_end: bool
    hours: float
    metrics: dict[str, float] = field(default_factory=dict)


def read_log(
    path: Path, limit: int | None = None, *, expected_epochs: int | None = None
) -> list[Step]:
    steps: list[Step] = []
    start = None
    with path.open() as stream:
        for raw in stream:
            line = re.sub(r"\x1b\[[\d;]*m", "", raw)
            if start is None and (stamp := TIME.search(line)):
                start = datetime.strptime(stamp[0], "%Y%m%d-%H:%M:%S.%f")
            if match := STEP.search(line):
                epoch, epochs, local, count, number, total = map(int, match.groups())
                if expected_epochs is not None and (
                    epochs != expected_epochs
                    or total != count * epochs
                    or not 1 <= local <= count
                    or number != (epoch - 1) * count + local
                ):
                    raise ValueError(
                        "Step header disagrees with the expected training horizon"
                    )
                if limit is not None and number > limit:
                    break
                stamp = TIME.search(line)
                if not stamp or start is None or number != len(steps) + 1:
                    raise ValueError(
                        f"Invalid timestamp or step sequence at step {number}"
                    )
                end = datetime.strptime(stamp[0], "%Y%m%d-%H:%M:%S.%f")
                steps.append(
                    Step(
                        epoch,
                        number,
                        local == count,
                        (end - start).total_seconds() / 3600,
                    )
                )
            elif steps and "│" in line:
                cells = [s.strip() for s in line.strip().strip("│").split("│")]
                for key, value in zip(cells[::2], cells[1::2]):
                    if key in METRICS or key.startswith("timeperf/"):
                        steps[-1].metrics[key] = float(value)
    if not steps:
        raise ValueError("No completed training steps found")
    if expected_epochs is not None and steps[-1].number != total:
        raise ValueError("Training ended before the final step")
    return steps


def observations(steps: list[Step], key: str) -> list[tuple[int, float]]:
    return [
        (s.number, s.metrics[key])
        for s in steps
        if key in s.metrics
        and (
            key != "timeperf/eval"
            or "eval-rollout/reward" in s.metrics
            or "sft-eval/loss/avg" in s.metrics
        )
    ]


def average(steps: list[Step], key: str) -> float | None:
    values = observations(steps, key)
    return mean(y for _, y in values) if values else None


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def plot(runs: dict[str, list[Step]], keys: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {"Current": ("#0057B8", "-"), "Baseline": ("#AD4600", "--")}
    columns = 4 if len(keys) == 4 else min(3, len(keys))
    rows = (len(keys) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(10, 1.65 * rows + 0.5)
    )
    ends = [s.number for s in runs["Current"] if s.epoch_end]
    for ax, key in zip(axes.flat, keys):
        unit = "Seconds" if key.startswith("timeperf/") else "Reward"
        if key in SFT_METRICS:
            unit = "Loss" if "/loss/" in key else "Perplexity"
        for label, steps in reversed(runs.items()):
            pairs = observations(steps, key)
            if not pairs:
                continue
            x, y = zip(*pairs, strict=True)
            if key.endswith("acceptance_rate"):
                unit, y = "Percent", [100 * v for v in y]
            elif key.endswith("mean_accepted_len"):
                unit = "Tokens"
            color, style = styles[label]
            if len(x) >= 5:
                ax.plot(x, y, color=color, ls=style, alpha=0.48, lw=1.0)
                ax.plot(
                    x[4:],
                    [mean(y[i - 4 : i + 1]) for i in range(4, len(y))],
                    color=color,
                    ls=style,
                    lw=2.2,
                )
            else:
                ax.plot(x, y, color=color, ls=style, marker="o")
        for end in ends:
            ax.axvline(end, color="gray", ls=":", alpha=0.4)
        if key == "timeperf/save" and all(
            y > 0 for steps in runs.values() for _, y in observations(steps, key)
        ):
            ax.set_yscale("log")
        ax.set(
            title=METRICS.get(key, key.removeprefix("timeperf/")),
            ylabel=unit,
            xticks=sorted({1, *ends}),
            xlim=(0, runs["Current"][-1].number + 1),
        )
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(keys) :]:
        ax.remove()
    handles = [
        Line2D([], [], color=styles[label][0], ls=styles[label][1], label=label)
        for label in runs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(runs), frameon=False)
    fig.supxlabel("Training step", fontsize=9, y=0.01)
    fig.tight_layout(
        rect=(0, 0.22 / fig.get_figheight(), 1, 1 - 0.32 / fig.get_figheight()),
        pad=0.45,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(
    runs: dict[str, list[Step]], chart_dir: Path, image_prefix: str, suite: str = "grpo"
) -> str:
    metrics = GRPO_METRICS if suite == "grpo" else SFT_METRICS
    sections = [
        f"# Ascend nightly · Geometry3K {suite.upper()}",
        "## Epoch results",
        "Elapsed hours include startup and validation. Training metrics are epoch means.",
    ]
    epoch_rows = []
    for epoch in sorted({s.epoch for s in runs["Current"] if s.epoch_end}):
        for label, steps in runs.items():
            group = [s for s in steps if s.epoch == epoch]
            if not group or not group[-1].epoch_end:
                continue
            previous = steps[group[0].number - 2].hours if group[0].number > 1 else 0
            row = [
                str(epoch),
                label,
                str(group[-1].number),
                f"{group[-1].hours - previous:.3f}",
                f"{group[-1].hours:.3f}",
            ]
            for key, fmt in [
                (k, ".2%" if k.endswith("rate") else ".5f") for k in metrics
            ] + [("timeperf/eval", ".2f")]:
                value = average(group, key)
                row.append("—" if value is None else format(value, fmt))
            epoch_rows.append(row)
    sections.append(
        table(
            [
                "Epoch",
                "Run",
                "Step",
                "Epoch (h)",
                "From start (h)",
                *metrics.values(),
                "Validation (s)",
            ],
            epoch_rows,
        )
    )
    timers = sorted(
        {
            k
            for steps in runs.values()
            for s in steps
            for k in s.metrics
            if k.startswith("timeperf/")
        }
    )
    main = [f"timeperf/{key}" for key in MAIN_TIMERS if f"timeperf/{key}" in timers]
    chart_dir.mkdir(parents=True, exist_ok=True)
    sections.extend(
        [
            "## Metrics",
            "Lines show individual steps and trailing five-step means. Dots mark sparse observations.",
        ]
    )
    for name, alt, keys in [
        ("quality", "Metrics", list(metrics)),
        ("training-times", "Training and validation timings", main),
        ("other-times", "Other timings", [k for k in timers if k not in main]),
    ]:
        if name == "training-times":
            sections.append("## Timings")
        if keys:
            plot(runs, keys, chart_dir / f"{name}.png")
            sections.append(f"![{alt}]({image_prefix.rstrip('/')}/{name}.png)")
    timing_rows = []
    for key in main + [k for k in timers if k not in main]:
        means = [average(steps, key) for steps in runs.values()]
        row = [
            key.removeprefix("timeperf/"),
            *["—" if v is None else f"{v:.5g}" for v in means],
        ]
        if "Baseline" in runs:
            current, baseline = means
            row.append(
                f"{current / baseline - 1:+,.1%}"
                if current is not None and baseline
                else "—"
            )
        timing_rows.append(row)
    headers = ["Timer", *[f"{label} (s)" for label in runs]]
    if "Baseline" in runs:
        headers.append("Change")
    sections.extend(
        [
            "### Mean seconds per observed step",
            "Evaluation includes steps that logged validation metrics.",
            table(headers, timing_rows),
        ]
    )
    return "\n\n".join(sections) + "\n"


def read_snapshot(path: Path) -> list[Step]:
    return [Step(**row) for row in json.loads(path.read_text())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["grpo", "sft"], default="grpo")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--current", type=Path, help="Current main.log")
    sources.add_argument("--snapshot", type=Path, help="Current metrics.json")
    parser.add_argument("--baseline-snapshot", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="Output Markdown file"
    )
    parser.add_argument(
        "--image-prefix", help="Published chart URL prefix for the Actions summary"
    )
    args = parser.parse_args()
    current = read_log(args.current) if args.current else read_snapshot(args.snapshot)

    runs = {"Current": current}
    if args.baseline:
        baseline = read_log(args.baseline, len(current))
        if [(s.epoch, s.epoch_end) for s in current] != [
            (s.epoch, s.epoch_end) for s in baseline
        ]:
            parser.error(
                "Baseline must cover the current steps with the same epoch boundaries"
            )
        runs["Baseline"] = baseline
    if args.baseline_snapshot:
        baseline = read_snapshot(args.baseline_snapshot)
        if [(s.number, s.epoch, s.epoch_end) for s in baseline] == [
            (s.number, s.epoch, s.epoch_end) for s in current
        ]:
            runs["Baseline"] = baseline
        else:
            print(
                "Baseline has a different training horizon. Reporting the current run."
            )
    charts = args.output.parent / args.output.stem
    markdown = report(runs, charts, args.image_prefix or charts.name, args.suite)
    args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
