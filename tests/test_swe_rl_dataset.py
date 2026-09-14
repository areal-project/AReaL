import json

from examples.swe.train_swe_rl import get_swe_dataset


def test_get_swe_dataset_preserves_fields_from_later_records(tmp_path) -> None:
    """Heterogeneous JSONL records retain fields absent from the first row."""
    # Arrange
    dataset_path = tmp_path / "swe.jsonl"
    records = [
        {"instance_id": "first", "problem_statement": "first problem"},
        {
            "instance_id": "second",
            "problem_statement": "second problem",
            "repo": "example/repo",
        },
    ]
    dataset_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    # Act
    dataset = get_swe_dataset(str(dataset_path), min_items=2)

    # Assert
    assert "repo" in dataset.column_names
    assert dataset[0]["repo"] is None
    assert dataset[1]["repo"] == "example/repo"
