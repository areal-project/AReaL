from __future__ import annotations

from typing import Any, Mapping, Optional


def extract_task_id(meta: Optional[Mapping[str, Any]]) -> Optional[Any]:
    """Extract a stable task identifier from metadata.

    Important: a valid task id can be `0` (e.g. sample_index "0"), so we must
    not use truthiness-based fallbacks like `a or b or c`.
    """
    if not meta:
        return None

    for key in ("task_id", "sample_index", "id"):
        if key not in meta:
            continue
        v = meta.get(key)
        if v is None:
            continue
        # Treat empty strings as missing, but keep "0".
        if isinstance(v, str) and v.strip() == "":
            continue
        return v

    return None

