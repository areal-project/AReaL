"""Tests for SGLang LoRA disk weight-update request building.

Regression: each train step loads `lora-<name>-v{N}` via /load_lora_adapter but
never unloads old versions, so sglang accumulates adapters until it hangs
(~130 steps, VRAM creeps up). The builder must emit a best-effort unload of the
version that falls outside the retention window (kept for off-policy rollouts).
"""

from areal.api.io_struct import WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend


def _lora_meta(version, keep, lora_name="lora-gsm8k", path="/tmp/wu"):
    return WeightUpdateMeta(
        type="disk",
        path=path,
        use_lora=True,
        lora_name=lora_name,
        lora_keep_versions=keep,
        version=version,
    )


def _triples(reqs):
    """(endpoint, lora_name, best_effort) for each request."""
    return [
        (r.endpoint, r.payload.get("lora_name"), r.best_effort) for r in reqs.requests
    ]


def test_disk_update_unloads_stale_version_beyond_window():
    """load v{N} also emits a best-effort unload of v{N - keep}."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(10, keep=4))
    triples = _triples(reqs)
    assert ("/load_lora_adapter", "lora-gsm8k-v10", False) in triples
    unloads = [t for t in triples if t[0] == "/unload_lora_adapter"]
    assert unloads == [("/unload_lora_adapter", "lora-gsm8k-v6", True)]


def test_disk_update_no_unload_within_window():
    """No unload when version - keep < 0 (nothing stale yet)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(2, keep=4))
    endpoints = [r.endpoint for r in reqs.requests]
    assert "/unload_lora_adapter" not in endpoints
    assert "/load_lora_adapter" in endpoints


def test_disk_update_keep_zero_disables_unload():
    """lora_keep_versions=0 keeps the old behaviour (load only)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(100, keep=0))
    assert [r.endpoint for r in reqs.requests] == ["/load_lora_adapter"]


def test_disk_update_unload_is_best_effort():
    """The unload request must be best-effort (stale adapter may be gone)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(10, keep=4))
    unload = next(r for r in reqs.requests if r.endpoint == "/unload_lora_adapter")
    assert unload.best_effort is True


def test_disk_update_full_model_unchanged():
    """Non-LoRA disk update path is untouched."""
    meta = WeightUpdateMeta(type="disk", path="/tmp/wu", use_lora=False)
    reqs = SGLangBackend().build_disk_weight_update_requests(meta)
    assert [r.endpoint for r in reqs.requests] == ["/update_weights_from_disk"]
