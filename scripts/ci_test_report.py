#!/usr/bin/env python3

"""Capture pytest outcomes and summarize CI test-count changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPORT_DIR_ENV = "AREAL_TEST_REPORT_DIR"

_selected_nodeids: list[str] = []
_outcomes: dict[str, str] = {}


def pytest_sessionstart(session) -> None:
    global _selected_nodeids, _outcomes
    _selected_nodeids = []
    _outcomes = {}


def pytest_collection_finish(session) -> None:
    global _selected_nodeids
    _selected_nodeids = [item.nodeid for item in session.items]
    _write_report()


def pytest_runtest_logreport(report) -> None:
    if report.failed:
        _outcomes[report.nodeid] = "failed"
    elif report.skipped and _outcomes.get(report.nodeid) != "failed":
        _outcomes[report.nodeid] = "skipped"
    elif report.when == "call" and report.passed:
        _outcomes[report.nodeid] = "passed"


def pytest_sessionfinish(session, exitstatus) -> None:
    _write_report(exitstatus=int(exitstatus))


def _write_report(exitstatus: int | None = None) -> None:
    report_dir = os.environ.get(REPORT_DIR_ENV)
    if not report_dir:
        return

    payload = {
        "exitstatus": exitstatus,
        "outcomes": _outcomes,
        "selected_nodeids": _selected_nodeids,
    }
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"pytest-{os.getpid()}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    selected: set[str] = set()
    outcomes: dict[str, str] = {}
    outcome_priority = {"passed": 1, "skipped": 2, "failed": 3}

    for report in reports:
        selected.update(report.get("selected_nodeids", []))
        for nodeid, outcome in report.get("outcomes", {}).items():
            if outcome_priority.get(outcome, 0) >= outcome_priority.get(
                outcomes.get(nodeid, ""), 0
            ):
                outcomes[nodeid] = outcome

    return {"outcomes": outcomes, "selected_nodeids": sorted(selected)}


def summarize_reports(
    current: dict[str, Any],
    base: dict[str, Any] | None = None,
    *,
    base_is_inventory: bool = False,
    changed_test_files: set[str] | None = None,
    unavailable_base_test_files: set[str] | None = None,
) -> dict[str, int]:
    selected = set(current.get("selected_nodeids", []))
    outcomes = current.get("outcomes", {})
    passed = {nodeid for nodeid, outcome in outcomes.items() if outcome == "passed"}
    failed = {nodeid for nodeid, outcome in outcomes.items() if outcome == "failed"}
    skipped = {nodeid for nodeid, outcome in outcomes.items() if outcome == "skipped"}
    executed = passed | failed
    incomplete = selected - executed - skipped

    counts = {
        "selected": len(selected),
        "executed": len(executed),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "incomplete": len(incomplete),
    }
    if base is None:
        return counts

    base_selected = set(base.get("selected_nodeids", []))
    unmatched_selected = selected - base_selected
    unavailable_selected: set[str] = set()
    if unavailable_base_test_files is not None:
        unavailable_selected = {
            nodeid
            for nodeid in selected
            if nodeid.split("::", maxsplit=1)[0] in unavailable_base_test_files
        }
    comparable_unmatched = unmatched_selected - unavailable_selected
    new_selected = comparable_unmatched
    if changed_test_files is not None:
        new_selected = {
            nodeid
            for nodeid in comparable_unmatched
            if nodeid.split("::", maxsplit=1)[0] in changed_test_files
        }
    counts.update(
        {
            "base_selected": len(selected & base_selected)
            if base_is_inventory
            else len(base_selected),
            "new_selected": len(new_selected),
            "new_executed": len(new_selected & executed),
            "new_skipped": len(new_selected & skipped),
            "new_incomplete": len(new_selected & incomplete),
        }
    )
    if changed_test_files is not None:
        counts["unchanged_unmatched"] = len(comparable_unmatched - new_selected)
    if unavailable_base_test_files is not None:
        counts["uncompared_selected"] = len(unavailable_selected)
    return counts


def render_summary(
    counts: dict[str, int],
    *,
    suite: str,
    base_sha: str | None = None,
    base_is_inventory: bool = False,
) -> str:
    rows = [
        (f"Selected {suite} test cases", counts["selected"]),
        (f"Executed {suite} test cases", counts["executed"]),
        ("Passed", counts["passed"]),
        ("Failed", counts["failed"]),
        ("Skipped", counts["skipped"]),
        ("Incomplete", counts["incomplete"]),
    ]
    if "base_selected" in counts:
        base_label = f" at `{base_sha[:12]}`" if base_sha else ""
        if base_is_inventory:
            previous_label = f"Selected cases present in base{base_label}"
        else:
            previous_label = f"Base selected cases{base_label}"
        rows.extend(
            [
                (previous_label, counts["base_selected"]),
                ("New test cases selected", counts["new_selected"]),
                ("New test cases executed", counts["new_executed"]),
                ("New test cases skipped", counts["new_skipped"]),
                ("New test cases incomplete", counts["new_incomplete"]),
            ]
        )
        if "unchanged_unmatched" in counts:
            rows.append(
                (
                    "Unmatched cases from unchanged test files",
                    counts["unchanged_unmatched"],
                )
            )
        if "uncompared_selected" in counts:
            rows.append(
                ("Cases without a base inventory", counts["uncompared_selected"])
            )

    lines = [f"## {suite} test summary", "", "| Metric | Count |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines) + "\n"


def _load_reports(path: Path) -> dict[str, Any]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return merge_reports(
        [json.loads(report.read_text(encoding="utf-8")) for report in paths]
    )


def _load_changed_test_files(path: Path) -> set[str]:
    return {
        line.strip().removeprefix("./")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--current", type=Path, required=True)
    summary_parser.add_argument("--base", type=Path)
    summary_parser.add_argument("--base-sha")
    summary_parser.add_argument("--base-is-inventory", action="store_true")
    summary_parser.add_argument("--changed-test-files", type=Path)
    summary_parser.add_argument("--unavailable-base-test-files", type=Path)
    summary_parser.add_argument("--suite", required=True)
    summary_parser.add_argument("--step-summary", type=Path)
    args = parser.parse_args()

    current = _load_reports(args.current)
    base = _load_reports(args.base) if args.base else None
    changed_test_files = (
        _load_changed_test_files(args.changed_test_files)
        if args.changed_test_files
        else None
    )
    unavailable_base_test_files = (
        _load_changed_test_files(args.unavailable_base_test_files)
        if args.unavailable_base_test_files
        else None
    )
    summary = render_summary(
        summarize_reports(
            current,
            base,
            base_is_inventory=args.base_is_inventory,
            changed_test_files=changed_test_files,
            unavailable_base_test_files=unavailable_base_test_files,
        ),
        suite=args.suite,
        base_sha=args.base_sha,
        base_is_inventory=args.base_is_inventory,
    )
    print(summary, end="")
    if args.step_summary:
        with args.step_summary.open("a", encoding="utf-8") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
