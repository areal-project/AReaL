import pytest

from examples.swe.arena_reward_breakdown import arena_reward_metrics


@pytest.mark.parametrize("schema", [None, True, "1", 2])
def test_unsupported_observation_schema_omits_measurements(schema):
    metrics = arena_reward_metrics(
        sub_scores=None,
        raw={"reward_observation": {"schema_version": schema, "diagnostics": {"x": 1}}},
        score=0.5,
    )
    assert metrics["reward_observation/present"] == 1
    assert metrics["reward_observation/valid"] == 0
    assert metrics["reward_observation/invalid_entries"] == 1
    assert "reward_observation/diagnostics/x" not in metrics


def test_observation_preserves_unicode_and_omits_missing_nonfinite_values():
    metrics = arena_reward_metrics(
        sub_scores={"外观": 0.7},
        raw={
            "reward_observation": {
                "schema_version": 1,
                "components": {
                    "外观/质量": {"value": 0.7, "weight": None, "share": float("nan")},
                    "bad": {"value": float("inf"), "weight": True},
                },
                "diagnostics": {"缺失": None, "有效": 2, "bad": float("-inf")},
            }
        },
        score=0.7,
    )
    assert metrics["reward_sub_score/外观"] == 0.7
    assert metrics["reward_observation/components/外观/质量/value"] == 0.7
    assert metrics["reward_observation/diagnostics/有效"] == 2
    assert metrics["reward_observation/valid"] == 1
    assert metrics["reward_observation/invalid_entries"] == 4
    measurements = {
        key
        for key in metrics
        if key.startswith(
            ("reward_observation/components/", "reward_observation/diagnostics/")
        )
    }
    assert measurements == {
        "reward_observation/components/外观/质量/value",
        "reward_observation/diagnostics/有效",
    }


def test_subscore_ids_reject_collisions_reserved_and_invalid_paths():
    metrics = arena_reward_metrics(
        sub_scores={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        raw={
            "reward_display": {
                "component_ids": {
                    "a": "quality/value",
                    "b": "quality/value",
                    "c": "invalid_entries",
                    "d": "bad//path",
                    "e": "bad\npath",
                }
            }
        },
        score=0.5,
    )
    assert metrics["reward_sub_score/quality/value"] == 1
    assert metrics["reward_sub_score/invalid_entries"] == 4
    assert sum(key.startswith("reward_sub_score/") for key in metrics) == 2
