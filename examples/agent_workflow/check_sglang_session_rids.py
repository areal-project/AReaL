# SPDX-License-Identifier: Apache-2.0

"""Check completed v1 session traces against native SGLang request logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_log_dir", type=Path)
    args = parser.parse_args()
    root = args.trial_log_dir

    uuids: set[str] = set()
    for trace_file in root.rglob("sessions-*.jsonl"):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            session_uuid = record.get("session_uuid")
            if isinstance(session_uuid, str):
                uuids.add(session_uuid)
    if not uuids:
        raise SystemExit("No session_uuid found in session tracer JSONL")

    matched: set[str] = set()
    rid_pattern = re.compile(r"\b([0-9a-f]{32}):[1-9][0-9]*\.[1-9][0-9]*\b")
    for log_file in root.rglob("*"):
        if log_file.suffix not in {".log", ".out", ".txt"}:
            continue
        try:
            with log_file.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    matched.update(
                        match.group(1)
                        for match in rid_pattern.finditer(line)
                        if match.group(1) in uuids
                    )
                    if matched:
                        break
        except OSError:
            continue
        if matched:
            break
    if not matched:
        raise SystemExit(
            "No SGLang RID matching a session UUID was found; "
            "enable native SGLang request logging for this check"
        )
    print(f"Matched {len(matched)} session UUID(s) to SGLang request IDs")


if __name__ == "__main__":
    main()
