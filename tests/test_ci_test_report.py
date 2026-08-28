from scripts.ci_test_report import merge_reports, render_summary, summarize_reports


def test_merge_reports_combines_parallel_pytest_processes():
    merged = merge_reports(
        [
            {
                "selected_nodeids": ["test_a", "test_b"],
                "outcomes": {"test_a": "passed", "test_b": "skipped"},
            },
            {
                "selected_nodeids": ["test_b", "test_c"],
                "outcomes": {"test_b": "failed", "test_c": "passed"},
            },
        ]
    )

    assert merged == {
        "selected_nodeids": ["test_a", "test_b", "test_c"],
        "outcomes": {
            "test_a": "passed",
            "test_b": "failed",
            "test_c": "passed",
        },
    }


def test_summarize_reports_counts_new_executed_cases():
    current = {
        "selected_nodeids": ["test_a", "test_b", "test_c", "test_d"],
        "outcomes": {
            "test_a": "passed",
            "test_b": "skipped",
            "test_c": "failed",
        },
    }
    base = {"selected_nodeids": ["test_a", "test_old"], "outcomes": {}}

    counts = summarize_reports(current, base)

    assert counts == {
        "selected": 4,
        "executed": 2,
        "passed": 1,
        "failed": 1,
        "skipped": 1,
        "incomplete": 1,
        "base_selected": 2,
        "new_selected": 3,
        "new_executed": 1,
        "new_skipped": 1,
        "new_incomplete": 1,
    }


def test_summarize_reports_uses_base_as_inventory():
    current = {
        "selected_nodeids": ["test_a", "test_new"],
        "outcomes": {"test_a": "passed", "test_new": "passed"},
    }
    base = {
        "selected_nodeids": ["test_a", "test_unselected_sibling"],
        "outcomes": {},
    }

    counts = summarize_reports(current, base, base_is_inventory=True)

    assert counts["base_selected"] == 1
    assert counts["new_selected"] == 1
    assert counts["new_executed"] == 1


def test_summarize_reports_ignores_unmatched_cases_from_unchanged_files():
    current = {
        "selected_nodeids": [
            "tests/test_npu.py::test_existing",
            "tests/test_npu.py::test_import_sensitive",
        ],
        "outcomes": {
            "tests/test_npu.py::test_existing": "passed",
            "tests/test_npu.py::test_import_sensitive": "passed",
        },
    }
    base = {
        "selected_nodeids": ["tests/test_npu.py::test_existing"],
        "outcomes": {},
    }

    counts = summarize_reports(
        current,
        base,
        base_is_inventory=True,
        changed_test_files={"tests/test_cpu.py"},
    )

    assert counts["new_selected"] == 0
    assert counts["new_executed"] == 0
    assert counts["unchanged_unmatched"] == 1


def test_summarize_reports_counts_unmatched_cases_from_changed_files():
    current = {
        "selected_nodeids": ["tests/test_npu.py::test_new"],
        "outcomes": {"tests/test_npu.py::test_new": "passed"},
    }
    base = {"selected_nodeids": [], "outcomes": {}}

    counts = summarize_reports(
        current,
        base,
        base_is_inventory=True,
        changed_test_files={"tests/test_npu.py"},
    )

    assert counts["new_selected"] == 1
    assert counts["new_executed"] == 1
    assert counts["unchanged_unmatched"] == 0


def test_summarize_reports_excludes_cases_without_base_inventory():
    current = {
        "selected_nodeids": ["tests/test_npu.py::test_import_sensitive"],
        "outcomes": {"tests/test_npu.py::test_import_sensitive": "passed"},
    }
    base = {"selected_nodeids": [], "outcomes": {}}

    counts = summarize_reports(
        current,
        base,
        base_is_inventory=True,
        changed_test_files={"tests/test_npu.py"},
        unavailable_base_test_files={"tests/test_npu.py"},
    )

    assert counts["new_selected"] == 0
    assert counts["new_executed"] == 0
    assert counts["unchanged_unmatched"] == 0
    assert counts["uncompared_selected"] == 1


def test_render_summary_includes_suite_and_base_comparison():
    summary = render_summary(
        {
            "selected": 10,
            "executed": 8,
            "passed": 8,
            "failed": 0,
            "skipped": 2,
            "incomplete": 0,
            "base_selected": 9,
            "new_selected": 2,
            "new_executed": 2,
            "new_skipped": 0,
            "new_incomplete": 0,
        },
        suite="NPU",
        base_sha="1234567890abcdef",
    )

    assert "## NPU test summary" in summary
    assert "| Selected NPU test cases | 10 |" in summary
    assert "| Base selected cases at `1234567890ab` | 9 |" in summary
    assert "| New test cases executed | 2 |" in summary
