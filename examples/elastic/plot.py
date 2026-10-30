# SPDX-License-Identifier: Apache-2.0

"""Plot measured demo events; never synthesize successful lifecycle transitions."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def plot_run(root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations = read_jsonl(root / "observations.jsonl")
    events = read_jsonl(root / "state/events.jsonl")
    decisions = read_jsonl(root / "state/decisions.jsonl")
    event_snapshots = not observations
    observations = observations or events
    if not observations or not events:
        raise ValueError("Plot requires real observations and lifecycle events")
    start = min(row["monotonic"] for row in events + observations)
    times = [row["monotonic"] - start for row in observations]
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    protocol_check = (root / "verification-kind.json").exists() and json.loads(
        (root / "verification-kind.json").read_text()
    ).get("kind") == "forced_restart_protocol_check"
    fig.suptitle(
        "Forced restart protocol · measured check"
        if protocol_check
        else "Elastic Actor lifecycle · measured run",
        fontsize=17,
        weight="bold",
    )
    ready = [
        sum(row["counts"].get(key, 0) for key in ("ready", "reserved"))
        for row in observations
    ]
    axes[0].step(times, ready, where="post", color="#2458a6", linewidth=2)
    axes[0].axhline(
        observations[0].get("batch_size", 4),
        color="#9ca3af",
        linestyle=":",
        label="one batch",
    )
    unit = (
        "task groups"
        if observations[0].get("count_unit") == "task_groups"
        else "episodes"
    )
    axes[0].set_ylabel(
        f"Unconsumed {unit}\n(event snapshots)"
        if event_snapshots
        else f"Unconsumed\nready/reserved {unit}"
    )
    axes[0].legend(loc="upper right")
    lifecycle = [e for e in events if e["state"] in {"STARTING", "STOPPED"}]
    axes[1].step(
        [e["monotonic"] - start for e in lifecycle]
        + [max(r["monotonic"] for r in events + observations) - start],
        [len(e["actor_devices"]) if e["state"] == "STARTING" else 0 for e in lifecycle]
        + [
            0
            if lifecycle[-1]["state"] == "STOPPED"
            else len(lifecycle[-1]["actor_devices"])
        ],
        where="post",
        color="#16836b",
        linewidth=2,
    )
    axes[1].set_ylabel("Actor GPU lease\n(request → confirmed release)")
    axes[1].set_yticks([0, 4])
    axes[1].set_ylim(-0.25, 4.6)
    released_path = root / "released.json"
    if protocol_check and released_path.exists():
        probes = json.loads(released_path.read_text()).get(
            "fresh_generation_while_actor_off", []
        )
        off = next(
            (
                e
                for e in lifecycle
                if e["state"] == "STOPPED" and e["applied_version"] > 0
            ),
            None,
        )
        if probes and off is not None:
            x = off["monotonic"] - start
            axes[1].scatter([x], [0], color="#d56736", s=45, zorder=3)
            axes[1].annotate(
                f"Actor released\n{len(probes)} replicas generated fresh tokens",
                xy=(x, 0),
                xytext=(x - 65, 1.5),
                fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#d56736"},
            )
    axes[2].step(
        times,
        [r["applied_version"] for r in observations],
        where="post",
        color="#9857b7",
        linewidth=2,
    )
    axes[2].set_ylabel("Training version")
    axes[2].set_xlabel("Elapsed seconds")
    for row in decisions:
        action = row["decision"]["action"]
        x = row["monotonic"] - start
        axes[1].axvline(x, color="#6b7280", alpha=0.35, linestyle="--")
        axes[1].annotate(
            "request " + action.removesuffix("_actor"),
            (x, 4.2),
            fontsize=8,
            rotation=30,
        )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    target = root / "timeline.svg"
    fig.savefig(target)
    fig.savefig(root / "timeline.png", dpi=160)
    plt.close(fig)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot_run(parser.parse_args().run)
