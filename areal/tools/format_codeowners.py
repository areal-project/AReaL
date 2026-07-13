#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that formats and lints .github/CODEOWNERS.

Behavior:
- Strips trailing whitespace and ensures a single trailing newline.
- Re-aligns the owners column so every rule line has the owners list starting
  at the same column (computed from the longest path in the file).
- Preserves trailing inline comments (e.g. ``/path @owner @owner2  # note``)
  verbatim, separated from the owners by two spaces.
- Validates that every owner token starts with '@' and contains only valid
  GitHub username/team characters.
- Errors on duplicate path patterns.
- Warns (does not fail) on rules with fewer than two owners, since the
  governance policy expects ownership to degrade gracefully when a single
  owner is unavailable.

Following the repo's other local hooks, the script exits non-zero if it had
to rewrite the file so CI flags the diff and the developer commits the fix.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CODEOWNERS_PATH = Path(".github/CODEOWNERS")
OWNER_RE = re.compile(
    r"^@[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:/[A-Za-z0-9._-]+)?$"
)
INLINE_COMMENT_RE = re.compile(r"\s#.*$")
MIN_COLUMN = 32


def format_codeowners(path: Path) -> int:
    """Return 0 if no changes needed, 1 if the file was rewritten, 2 on error."""
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    original = path.read_text(encoding="utf-8")
    raw_lines = original.splitlines()

    parsed: list[tuple[str, ...]] = []
    longest_path = 0
    errors: list[str] = []

    for lineno, line in enumerate(raw_lines, 1):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            parsed.append(("raw", stripped))
            continue

        # Split off any trailing inline comment (whitespace then '#'); the
        # comment is preserved verbatim and re-emitted after the owners.
        m = INLINE_COMMENT_RE.search(stripped)
        if m:
            inline_comment = m.group(0).strip()
            rule_part = stripped[: m.start()].rstrip()
        else:
            inline_comment = ""
            rule_part = stripped

        tokens = rule_part.split()
        if len(tokens) < 2:
            errors.append(f"line {lineno}: rule has no owners: {stripped!r}")
            continue

        path_pat, *owners = tokens
        for owner in owners:
            if not OWNER_RE.match(owner):
                errors.append(f"line {lineno}: invalid owner token {owner!r}")

        parsed.append(("rule", path_pat, tuple(owners), inline_comment, lineno))
        longest_path = max(longest_path, len(path_pat))

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    column = max(MIN_COLUMN, ((longest_path + 5) // 4) * 4)

    seen_paths: dict[str, int] = {}
    rendered: list[str] = []
    duplicate_errors: list[str] = []
    single_owner_warnings: list[str] = []

    for entry in parsed:
        if entry[0] == "raw":
            rendered.append(entry[1])
            continue

        _, path_pat, owners, inline_comment, lineno = entry
        if path_pat in seen_paths:
            duplicate_errors.append(
                f"line {lineno}: duplicate path pattern {path_pat!r} "
                f"(first defined on line {seen_paths[path_pat]})"
            )
            continue
        seen_paths[path_pat] = lineno

        if len(owners) < 2:
            single_owner_warnings.append(
                f"line {lineno}: {path_pat} has only one owner ({owners[0]}); "
                f"governance expects >=2 to avoid single points of failure"
            )

        rule_line = f"{path_pat.ljust(column)}{' '.join(owners)}"
        if inline_comment:
            rule_line = f"{rule_line}  {inline_comment}"
        rendered.append(rule_line)

    if duplicate_errors:
        for err in duplicate_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    for warn in single_owner_warnings:
        print(f"WARN: {warn}", file=sys.stderr)

    new = "\n".join(rendered).rstrip("\n") + "\n"
    if new != original:
        path.write_text(new, encoding="utf-8")
        print(f"Rewrote {path} (column {column})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(format_codeowners(CODEOWNERS_PATH))
