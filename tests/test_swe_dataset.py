import json

from examples.swe.train_swe_rl import get_swe_dataset


def test_swe_dataset_preserves_fields_missing_from_first_record(tmp_path):
    """Later optional fields remain available after Arrow conversion."""
    records = [
        {
            "instance_id": "repo__project-1",
            "problem_statement": "first",
            "eval_script": "pytest",
        },
        {
            "instance_id": "repo__project-2",
            "problem_statement": "second",
            "eval_script": "pytest",
            "repo_language": "python",
        },
    ]
    dataset_path = tmp_path / "swe.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    dataset = get_swe_dataset(str(dataset_path), min_items=2)

    assert "repo_language" in dataset.column_names
    assert dataset[0]["repo_language"] is None
    assert dataset[1]["repo_language"] == "python"
