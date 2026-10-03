"""Summarize GSM8K evaluation by question from full-dataset run artifacts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def question_results(run: Path, step: int) -> dict[str, int]:
    roots = list((run / "results/logs/root").glob("*/trial0/eval-rollout"))
    assert len(roots) == 1, (run, roots)
    files = sorted((roots[0] / str(step)).glob("*.jsonl"))
    assert len(files) == 1319, (run, step, len(files))
    results = {}
    for file in files:
        rows = [json.loads(line) for line in file.open()]
        assert len(rows) == 4, (file, len(rows))
        prompts = {row["prompt"] for row in rows}
        assert len(prompts) == 1, file
        rewards = [row["original_reward"] for row in rows]
        assert all(reward in (0, 1) for reward in rewards), file
        key = hashlib.sha256(next(iter(prompts)).encode()).hexdigest()
        assert key not in results, file
        results[key] = int(sum(rewards))
    return results


def summarize(root: Path) -> dict:
    all_results = {
        optimizer: {
            epoch: question_results(root / optimizer, 29 * epoch)
            for epoch in range(1, 4)
        }
        for optimizer in ("adamw", "muon")
    }
    question_ids = set(all_results["adamw"][1])
    assert all(
        set(results) == question_ids
        for optimizer_results in all_results.values()
        for results in optimizer_results.values()
    )
    report = {"questions": len(question_ids), "samples_per_question": 4, "epochs": {}}
    for epoch in range(1, 4):
        a = all_results["adamw"][epoch]
        m = all_results["muon"][epoch]
        report["epochs"][epoch] = {}
        for name, results in (("adamw", a), ("muon", m)):
            hist = Counter(results.values())
            report["epochs"][epoch][name] = {
                "correct_per_question_histogram": {
                    str(correct): hist[correct] for correct in range(5)
                },
                "correct_answers": sum(results.values()),
                "total_answers": 4 * len(results),
                "mean_reward": sum(results.values()) / (4 * len(results)),
                "pass_at_4": sum(value > 0 for value in results.values())
                / len(results),
                "all_4_correct": hist[4] / len(results),
            }
        differences = Counter(m[key] - a[key] for key in question_ids)
        report["epochs"][epoch]["paired_question_difference"] = {
            str(diff): differences[diff] for diff in range(-4, 5)
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    report = summarize(root)
    (root / "eval_detail.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
