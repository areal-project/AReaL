"""SQL-structure-level failure abstraction for LLB-DB Region FS.

The ALFWorld Region-FS gain comes from replacing instance entities with shared,
actionable rules. This module provides the DB analogue: schema/table/column names
are discarded and reflection bullets are mapped to SQL reasoning guardrails.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple
import re


def db_signature(skill_list: Sequence[str]) -> Tuple[str, str, Tuple[str, ...]]:
    skills = set(skill_list or [])
    if "insert" in skills:
        op = "insert"
    elif "update" in skills:
        op = "update"
    elif "delete" in skills:
        op = "delete"
    else:
        op = "select"
    if op != "select":
        shape = "mutation"
    elif skills & {"subquery_nested", "subquery_multiple", "subquery_single"}:
        shape = "subquery"
    elif skills & {
        "group_by_single_column", "group_by_multiple_columns",
        "having_single_condition_with_aggregate", "having_multiple_conditions_with_aggregate",
        "having_aggregate_calculation",
    }:
        shape = "aggregate"
    else:
        shape = "simple"
    groups = {
        "predicate": {"where_single_condition", "where_multiple_conditions", "where_nested_conditions"},
        "group_having": {"group_by_single_column", "group_by_multiple_columns", "having_single_condition_with_aggregate", "having_multiple_conditions_with_aggregate", "having_aggregate_calculation"},
        "ordering": {"order_by_single_column", "order_by_multiple_columns_same_direction", "order_by_multiple_columns_different_directions"},
        "pagination": {"limit_only", "limit_and_offset"},
        "aliasing": {"column_alias", "table_alias"},
        "nested": {"subquery_nested", "subquery_multiple", "subquery_single"},
    }
    mods = tuple(sorted(name for name, members in groups.items() if skills & members))
    return op, shape, mods


_RULES = {
    "group_having": "Keep row filters in WHERE and aggregate/group filters in HAVING; group every non-aggregated output column.",
    "subquery_scope": "Check subquery scope and correlation: compute the requested comparison population before filtering the outer rows.",
    "aggregate": "Verify aggregate function, grouping level, and whether the comparison is per-group or global; avoid double aggregation.",
    "predicate": "Preserve AND/OR precedence with parentheses and translate every requested boundary and string condition exactly.",
    "ordering": "Use the requested ORDER BY columns and directions exactly; preserve database row order in Final Answer.",
    "pagination": "Apply filtering/grouping first, then ORDER BY, then LIMIT/OFFSET; OFFSET is zero-based row skipping.",
    "distinct": "Check whether joins/grouping create duplicate rows and use DISTINCT only when the requested result is logically unique.",
    "projection": "Return exactly the requested columns in the requested order; do not substitute similarly named IDs or aliases.",
    "mutation": "For INSERT/UPDATE/DELETE, derive affected rows dynamically, preserve exact literals/types, execute once, then verify the changed state.",
    "precision": "Preserve decimal precision and exact string casing; do not round, truncate, or rewrite returned database values.",
    "protocol": "Use the exact Action directive and submit the database result as comma-separated tuples with no extra prose.",
}


def classify_failure_text(text: str) -> List[str]:
    t = " ".join((text or "").lower().split())
    labels = []
    def hit(label, *patterns):
        if any(re.search(p, t) for p in patterns):
            labels.append(label)
    hit("group_having", r"\bhaving\b", r"\bgroup by\b", r"grouping")
    hit("subquery_scope", r"subquer", r"outer query", r"correlat", r"overall average", r"global average")
    hit("aggregate", r"\b(sum|avg|average|count|min|max)\b", r"aggregate", r"total")
    hit("predicate", r"\bwhere\b", r"condition", r"filter", r"\band\b", r"\bor\b", r"boundary")
    hit("ordering", r"order by", r"row order", r"ascending", r"descending", r"unordered", r"sequence")
    hit("pagination", r"\blimit\b", r"\boffset\b", r"skip(?:ping)?", r"pagination")
    hit("distinct", r"duplicate", r"\bdistinct\b", r"extra rows", r"missing rows", r"row count")
    hit("projection", r"wrong column", r"column order", r"requested columns", r"client_id", r"customer_id", r"alias")
    hit("mutation", r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"affected rows", r"changed state")
    hit("precision", r"decimal", r"precision", r"round", r"truncat", r"casing", r"case-sensitive", r"exact string")
    hit("protocol", r"action: answer", r"action: operation", r"protocol", r"directive", r"tuple format", r"extra prose")
    return labels


def build_structured_db_summary(
    failure_texts: Iterable[str], target_skills: Sequence[str], *, top_n: int = 4
) -> str:
    texts = [str(x) for x in failure_texts if x]
    if not texts:
        return ""
    op, shape, modifiers = db_signature(target_skills)
    counts = Counter(label for text in texts for label in set(classify_failure_text(text)))

    # Target-shape priors ensure the summary remains actionable even when noisy
    # reflection wording fails to mention the SQL construct explicitly.
    preferred = []
    if shape == "subquery": preferred += ["subquery_scope", "aggregate"]
    if shape == "aggregate": preferred += ["group_having", "aggregate"]
    if op != "select": preferred += ["mutation", "precision"]
    if "predicate" in modifiers: preferred.append("predicate")
    if "group_having" in modifiers: preferred.append("group_having")
    if "ordering" in modifiers: preferred.append("ordering")
    if "pagination" in modifiers: preferred.append("pagination")
    if "aliasing" in modifiers: preferred.append("projection")
    preferred += [label for label, _ in counts.most_common()]

    chosen = []
    for label in preferred:
        if label not in chosen and label in _RULES and label != "protocol":
            chosen.append(label)
        if len(chosen) >= top_n:
            break
    if not chosen:
        chosen = ["predicate", "projection"]

    pattern = f"operation={op}; shape={shape}; modifiers={','.join(modifiers) if modifiers else 'none'}"
    lines = [
        f"SQL failure guardrails for the current query pattern ({pattern}).",
        f"Derived from {len(texts)} compatible failed attempts in the selected task region.",
        "CHECK BEFORE EXECUTION:",
    ]
    lines.extend(f"- {_RULES[label]}" for label in chosen)
    lines += [
        "CHECK BEFORE FINAL ANSWER:",
        "- Execute the final SQL, copy exactly the returned rows/values, and do not infer or manually reorder results.",
        "- Use the required Action directive and comma-separated tuple format; no headers or explanatory prose.",
    ]
    return "\n".join(lines)
