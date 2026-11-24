"""Verify the completed nightly training run and export only its performance data."""

import argparse
import getpass
import json
from dataclasses import asdict
from pathlib import Path

from safetensors import safe_open

from scripts.ascend_nightly_grpo_report import GRPO_METRICS, SFT_METRICS, read_log


def verify_checkpoint(checkpoint: Path) -> None:
    for name in ("config.json", "tokenizer_config.json"):
        if not json.loads((checkpoint / name).read_text()):
            raise ValueError(f"Empty checkpoint metadata: {name}")
    weights = {}
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weights:
                    raise ValueError(f"Duplicate checkpoint weight: {key}")
                weights[key] = shard.name
    if not weights:
        raise ValueError("No checkpoint weights")
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        if json.loads(index.read_text())["weight_map"] != weights:
            raise ValueError("Checkpoint index does not match saved weights")
    elif set(weights.values()) != {"model.safetensors"}:
        raise ValueError("Missing sharded checkpoint index")


def verify_run(root: Path, experiment: str, suite: str, output: Path) -> None:
    run = Path(getpass.getuser()) / experiment / "nightly"
    steps = read_log(root / "logs" / run / "main.log", expected_epochs=2)
    ends = [s for s in steps if s.epoch_end]
    if [s.epoch for s in ends] != [1, 2] or steps[-1] != ends[-1]:
        raise ValueError("Expected two complete epochs")
    final = ends[-1]
    size = final.number - ends[0].number
    if size != ends[0].number:
        raise ValueError("Unexpected number of steps in epoch 2")
    metrics = GRPO_METRICS if suite == "grpo" else SFT_METRICS
    for key in metrics:
        expected = ends if "eval" in key else steps
        if any(key not in s.metrics for s in expected):
            raise ValueError(f"Missing nightly metric {key}")
    checkpoint_root = root / "checkpoints" / run / "default"
    checkpoint = (
        checkpoint_root / f"epoch1epochstep{size - 1}globalstep{final.number - 1}"
    )
    if list(checkpoint_root.glob("epoch*")) != [checkpoint]:
        raise ValueError("Expected a single checkpoint at the end of epoch 2")
    verify_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(s) for s in steps], allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("experiment")
    parser.add_argument("suite", choices=["grpo", "sft"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify_run(args.root, args.experiment, args.suite, args.output)
