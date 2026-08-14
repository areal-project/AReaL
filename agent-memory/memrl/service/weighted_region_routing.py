"""Pure helpers for weighted multi-label Region routing."""
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def normalize_target_weights(target_weights: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    cleaned = [(str(st), max(0.0, float(w))) for st, w in target_weights if st]
    total = sum(w for _, w in cleaned)
    return [(st, w / total) for st, w in cleaned] if total > 0 else []


def rank_regions_weighted(regions: Iterable[Any], target_weights: Sequence[Tuple[str, float]], min_count: float = 0.0):
    weights = normalize_target_weights(target_weights)
    ranked = []
    if not weights:
        return ranked
    for region in regions:
        utilities = getattr(region, "utility_by_subtask", {}) or {}
        counts = getattr(region, "counts_by_subtask", {}) or {}
        weighted_count = sum(w * float(counts.get(st, 0.0) or 0.0) for st, w in weights)
        if weighted_count < float(min_count):
            continue
        score = sum(w * float(utilities.get(st, 0.5)) for st, w in weights)
        ranked.append((score, weighted_count, region))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _metadata_dict(candidate: Dict[str, Any]) -> Dict[str, Any]:
    md = candidate.get("metadata") or {}
    if isinstance(md, dict):
        return md
    extra = getattr(md, "model_extra", None)
    if isinstance(extra, dict):
        return extra
    try:
        dumped = md.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    except Exception:
        return {}


def candidate_diversity_key(candidate: Dict[str, Any]):
    """Stable semantic dedup key: source task first, normalized text fallback."""
    md = _metadata_dict(candidate)
    task_id = md.get("task_id", md.get("sample_index"))
    benchmark = md.get("source_benchmark", "")
    if task_id is not None:
        return ("task", str(benchmark), str(task_id))
    text = candidate.get("content") or candidate.get("memory") or ""
    if not text:
        item = candidate.get("memory_item")
        text = getattr(item, "memory", "") if item is not None else ""
    normalized = " ".join(str(text).lower().split())
    return ("text", normalized[:1000]) if normalized else ("memory", str(candidate.get("memory_id")))


def apply_region_quota(global_ranked: Sequence[Dict[str, Any]], candidate_pool: Sequence[Dict[str, Any]], member_ids: Iterable[str], *, quota: int, sim_floor: float, k: int):
    member_ids = set(member_ids)
    quota = max(0, min(int(quota), int(k)))
    pool = sorted(candidate_pool, key=lambda c: float(c.get("score", 0.0)), reverse=True)
    picks = []
    seen_ids = set()
    seen_keys = set()

    def add_if_diverse(cand, target):
        mid = cand.get("memory_id")
        key = candidate_diversity_key(cand)
        if not mid or mid in seen_ids or key in seen_keys:
            return False
        target.append(cand)
        seen_ids.add(mid)
        seen_keys.add(key)
        return True

    for cand in pool:
        mid = cand.get("memory_id")
        if not mid or mid not in member_ids:
            continue
        if float(cand.get("similarity", 0.0)) < float(sim_floor):
            continue
        if add_if_diverse(cand, picks) and len(picks) >= quota:
            break

    fill = []
    # Preserve the original global ranking first, then use the larger pool to
    # replace semantic duplicates so the final context keeps k diverse memories.
    for source in (global_ranked, pool):
        for cand in source:
            add_if_diverse(cand, fill)
            if len(picks) + len(fill) >= int(k):
                return picks + fill, picks
    return picks + fill, picks


def dedupe_ranked_candidates(primary: Sequence[Dict[str, Any]], candidate_pool: Sequence[Dict[str, Any]], *, k: int):
    """Keep ranking priority while enforcing unique source-task and content keys."""
    result = []
    seen_ids = set()
    seen_keys = set()
    for source in (primary, candidate_pool):
        for cand in source:
            mid = cand.get("memory_id")
            key = candidate_diversity_key(cand)
            if not mid or mid in seen_ids or key in seen_keys:
                continue
            result.append(cand)
            seen_ids.add(mid)
            seen_keys.add(key)
            if len(result) >= int(k):
                return result
    return result
