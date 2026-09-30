# SPDX-License-Identifier: Apache-2.0
"""Summarize completed validation logs without conflating dumps with training."""

import argparse
import json
import re
from pathlib import Path


def summarize(directory: Path) -> dict:
    feedback = json.loads((directory / "feedback.json").read_text())
    if not feedback.get("training_completed"):
        raise ValueError("The training job did not complete successfully")
    results = {}
    for label in ("disabled", "enabled"):
        log = (directory / f"{label}.log").read_text()
        steps = []
        for line in log.splitlines():
            match = re.search(r"Train step (\d+)/\d+ done", line)
            if match:
                steps.append({"step": int(match[1])})
            if steps and "│" in line:
                fields = [part.strip() for part in line.split("│")[1:-1]]
                for key, value in zip(fields[::2], fields[1::2], strict=True):
                    if key.startswith("stale_") or key in (
                        "ppo_actor/update/stale_empty_minibatch_fraction",
                        "ppo_actor/update/grad_norm",
                    ):
                        steps[-1][key] = float(value)
        if [step["step"] for step in steps] != list(range(1, 9)):
            raise ValueError(f"{label}: expected all eight training steps")
        generated = sum(s.get("stale_generated_tokens", 0) for s in steps)
        masked = sum(s.get("stale_masked_tokens", 0) for s in steps)
        partial = sum(s.get("stale_partially_masked_trajectories", 0) for s in steps)
        if label == "enabled" and not (generated > masked > 0 and partial > 0):
            raise ValueError("Need nonzero masking and retained mixed training rows")
        if label == "disabled" and any(
            key.startswith("stale_") for step in steps for key in step
        ):
            raise ValueError("Disabled mode unexpectedly emitted masking metrics")
        dumps = list((directory / label).glob("logs/**/rollout/*/*.jsonl"))
        records = mixed = 0
        example = None
        for path in dumps:
            for line in path.read_text().splitlines():
                record = json.loads(line)
                records += 1
                runs = record.get("version_rle", [])
                if len(runs) > 1:
                    mixed += 1
                    if example is None:
                        example = {
                            "file": str(path.relative_to(directory)),
                            "head_version": record.get("head_version"),
                            "tail_version": record.get("tail_version"),
                            "version_rle": runs,
                        }
        results[label] = {
            "steps": steps,
            "generated_training_tokens": int(generated) if generated else None,
            "masked_training_tokens": int(masked) if generated else None,
            "masked_training_ratio": masked / generated if generated else None,
            "partially_masked_training_trajectories": int(partial)
            if generated
            else None,
            "dumped_records_including_unconsumed": records,
            "mixed_version_dumped_records_including_unconsumed": mixed,
            "mixed_version_dump_example": example,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory), indent=2) + "\n")
