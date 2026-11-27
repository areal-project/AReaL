import pytest

from areal.dataset.validation import (
    RecordLocation,
    validate_records,
)


def _report(records, *, profile, **kwargs):
    located_records = [
        (record, RecordLocation(kind="row", number=index + 1))
        for index, record in enumerate(records)
    ]
    return validate_records(located_records, profile=profile, **kwargs)


def test_text_rl_valid_record_with_required_answer_returns_valid():
    record = {
        "messages": [
            {"role": "system", "content": "Solve the problem."},
            {"role": "user", "content": "1 + 1 = ?"},
            {"role": "assistant", "content": "2"},
        ],
        "answer": "",
    }

    report = _report(
        [record],
        profile="text-rl",
        require_fields=["answer"],
    )

    assert report.checked == 1
    assert report.valid == 1
    assert report.invalid == 0
    assert report.diagnostics == ()
    assert not report.diagnostics_truncated


def test_text_rl_required_answer_uses_presence_not_truthiness():
    record = {
        "messages": [{"role": "user", "content": "1 + 1 = ?"}],
        "answer": None,
    }

    report = _report(
        [record],
        profile="text-rl",
        require_fields=["answer"],
    )

    assert report.valid == 1
    assert report.diagnostics == ()


def test_require_fields_generator_is_normalized_before_scanning():
    records = [
        {"messages": [{"role": "user", "content": "1 + 1 = ?"}]},
        {"messages": [{"role": "user", "content": "2 + 2 = ?"}]},
    ]

    def required_fields():
        yield "answer"

    report = _report(
        records,
        profile="text-rl",
        require_fields=required_fields(),
    )

    assert report.checked == 2
    assert report.invalid == 2
    assert [diagnostic.location.number for diagnostic in report.diagnostics] == [1, 2]


def test_text_rl_missing_required_answer_is_invalid():
    record = {"messages": [{"role": "user", "content": "1 + 1 = ?"}]}

    report = _report(
        [record],
        profile="text-rl",
        require_fields=["answer"],
    )

    assert report.invalid == 1
    assert report.diagnostics[0].field == "answer"
    assert report.diagnostics[0].code == "missing_field"


@pytest.mark.parametrize(
    ("record", "field", "code"),
    [
        ({}, "messages", "missing_field"),
        ({"messages": []}, "messages", "empty_list"),
        ({"messages": {}}, "messages", "invalid_type"),
    ],
)
def test_text_rl_invalid_messages_shape_is_diagnosed(record, field, code):
    report = _report([record], profile="text-rl")

    assert report.invalid == 1
    assert report.diagnostics[0].field == field
    assert report.diagnostics[0].code == code


def test_text_rl_message_errors_are_ordered_and_record_counts_once():
    record = {
        "messages": [
            {"role": "tool", "content": 3},
            "invalid-message",
        ]
    }

    report = _report([record], profile="text-rl")

    assert report.checked == 1
    assert report.invalid == 1
    assert [
        (diagnostic.field, diagnostic.code) for diagnostic in report.diagnostics
    ] == [
        ("messages[0].role", "unsupported_role"),
        ("messages[0].content", "invalid_type"),
        ("messages[1]", "invalid_type"),
    ]


def test_non_object_record_is_invalid():
    report = _report(["not-an-object"], profile="text-rl")

    assert report.invalid == 1
    assert report.diagnostics[0].field == "record"
    assert report.diagnostics[0].code == "invalid_type"
    assert report.diagnostics[0].location.number == 1


def test_tokenized_sft_valid_boolean_mask_returns_valid():
    record = {"input_ids": [1, 2, 3], "loss_mask": [False, True, True]}

    report = _report([record], profile="tokenized-sft")

    assert report.checked == 1
    assert report.valid == 1
    assert report.diagnostics == ()


def test_tokenized_sft_accepts_explicit_numeric_zero_and_one_mask():
    record = {"input_ids": [1, 2], "loss_mask": [0.0, 1.0]}

    report = _report([record], profile="tokenized-sft")

    assert report.valid == 1


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {},
            [
                ("input_ids", "missing_field"),
                ("loss_mask", "missing_field"),
            ],
        ),
        (
            {"input_ids": [], "loss_mask": []},
            [
                ("input_ids", "empty_list"),
                ("loss_mask", "empty_list"),
            ],
        ),
        (
            {"input_ids": {}, "loss_mask": []},
            [
                ("input_ids", "invalid_type"),
                ("loss_mask", "empty_list"),
            ],
        ),
        (
            {"input_ids": [1, 2], "loss_mask": [True]},
            [("loss_mask", "length_mismatch")],
        ),
    ],
)
def test_tokenized_sft_missing_empty_and_type_errors_are_diagnosed(record, expected):
    report = _report([record], profile="tokenized-sft")

    assert report.invalid == 1
    assert [
        (diagnostic.field, diagnostic.code) for diagnostic in report.diagnostics
    ] == expected


def test_tokenized_sft_token_bool_is_not_accepted_as_integer():
    record = {"input_ids": [True, -1, "2"], "loss_mask": [False, True, True]}

    report = _report([record], profile="tokenized-sft")

    assert report.invalid == 1
    assert [
        (diagnostic.field, diagnostic.code) for diagnostic in report.diagnostics
    ] == [
        ("input_ids[0]", "invalid_token_id"),
        ("input_ids[1]", "invalid_token_id"),
        ("input_ids[2]", "invalid_token_id"),
    ]


def test_tokenized_sft_invalid_and_all_zero_mask_are_diagnosed():
    record = {"input_ids": [1, 2, 3], "loss_mask": [0, 2, "1"]}

    report = _report([record], profile="tokenized-sft")

    assert report.invalid == 1
    assert [
        (diagnostic.field, diagnostic.code) for diagnostic in report.diagnostics
    ] == [
        ("loss_mask[1]", "invalid_loss_mask"),
        ("loss_mask[2]", "invalid_loss_mask"),
        ("loss_mask", "all_loss_mask_zero"),
    ]


@pytest.mark.parametrize(
    ("length", "max_length", "valid"),
    [(2, 2, True), (3, 2, False)],
)
def test_tokenized_sft_max_length_uses_inclusive_limit(length, max_length, valid):
    record = {
        "input_ids": list(range(length)),
        "loss_mask": [1] * length,
    }

    report = _report(
        [record],
        profile="tokenized-sft",
        max_length=max_length,
    )

    assert report.valid == int(valid)
    assert report.invalid == int(not valid)
    if not valid:
        assert report.diagnostics[0].code == "max_length_exceeded"


def test_tokenized_sft_multiple_errors_count_record_once():
    record = {"input_ids": [True, -1], "loss_mask": [0, 2]}

    report = _report([record], profile="tokenized-sft")

    assert report.checked == 1
    assert report.invalid == 1
    assert len(report.diagnostics) >= 3


def test_validate_records_limits_diagnostics_without_limiting_scan():
    records = [{"input_ids": [False], "loss_mask": [False]} for _ in range(20)]

    report = _report(
        records,
        profile="tokenized-sft",
        max_diagnostics=2,
    )

    assert report.checked == 20
    assert report.invalid == 20
    assert len(report.diagnostics) == 2
    assert report.diagnostics_truncated


def test_validate_records_does_not_truncate_when_count_reaches_limit_exactly():
    records = [{"input_ids": [False], "loss_mask": [True]} for _ in range(2)]

    report = _report(
        records,
        profile="tokenized-sft",
        max_diagnostics=2,
    )

    assert report.invalid == 2
    assert len(report.diagnostics) == 2
    assert not report.diagnostics_truncated


def test_validate_records_truncates_when_count_exceeds_limit():
    records = [{"input_ids": [False], "loss_mask": [True]} for _ in range(3)]

    report = _report(
        records,
        profile="tokenized-sft",
        max_diagnostics=2,
    )

    assert report.checked == 3
    assert report.invalid == 3
    assert len(report.diagnostics) == 2
    assert report.diagnostics_truncated


def test_report_dict_has_stable_shape():
    record = {"input_ids": [True], "loss_mask": [False]}

    report = _report([record], profile="tokenized-sft", max_diagnostics=5)

    assert report.to_dict() == {
        "profile": "tokenized-sft",
        "checked": 1,
        "valid": 0,
        "invalid": 1,
        "diagnostics_truncated": False,
        "diagnostics": [
            {
                "location": {"kind": "row", "number": 1},
                "field": "input_ids[0]",
                "code": "invalid_token_id",
                "message": "Expected a non-negative integer.",
            },
            {
                "location": {"kind": "row", "number": 1},
                "field": "loss_mask",
                "code": "all_loss_mask_zero",
                "message": "Expected at least one supervised position.",
            },
        ],
    }
