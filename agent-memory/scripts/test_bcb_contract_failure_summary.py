#!/usr/bin/env python3
"""Focused regression checks for BCB contract-aware failure-summary gating."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

# The submit host intentionally lacks the full MemOS runtime. Load the target
# runner with narrow stubs so this unit test exercises the contract-gate code
# itself without installing or modifying runtime dependencies.
def _load_runner_class():
    bcb_runner = ModuleType("memrl.run.bcb_runner")
    bcb_runner.BCBRunner = type("BCBRunner", (), {})
    bcb_runner.BCBSelection = type("BCBSelection", (), {})
    region_service = ModuleType("memrl.service.region_memory_service")
    region_service.RegionMemoryService = type("RegionMemoryService", (), {})
    hierarchy = ModuleType("memrl.configs.task_hierarchy")
    hierarchy.get_primary_subtask = lambda *_args, **_kwargs: "bcb/General"
    region_manager = ModuleType("memrl.service.region_manager")

    class RegionManager:
        @staticmethod
        def _parse_failure_fields(content):
            lines = content.splitlines()
            mode = next((line.split(":", 1)[1].strip() for line in lines
                         if line.startswith("FAILURE_MODE:")), "")
            return {"failure_mode": mode,
                    "mistakes": [line[2:] for line in lines if line.startswith("- ")],
                    "fixes": [], "avoids": []}

        @staticmethod
        def _format_failure_summary(fields, top_n=3):
            return f"Common failure patterns ({len(fields)} compatible failures)"

    region_manager.RegionManager = RegionManager
    sys.modules.update({
        "memrl.run.bcb_runner": bcb_runner,
        "memrl.service.region_memory_service": region_service,
        "memrl.configs.task_hierarchy": hierarchy,
        "memrl.service.region_manager": region_manager,
    })
    source = Path(__file__).resolve().parents[1] / "memrl/run/bcb_region_runner.py"
    spec = spec_from_file_location("_contract_gate_runner", source)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.BCBRegionRunner


BCBRegionRunner = _load_runner_class()


def _contract(task_id, prompt):
    return BCBRegionRunner._extract_bcb_task_contract(
        {"task_id": task_id, "prompt": prompt}
    )


def _failure(task_id, prompt, failure_mode="Output format mismatch"):
    return {
        "metadata": {
            "task_id": task_id,
            "outcome": "failure",
            "success": False,
            "task_description": prompt,
            "full_content": (
                f"FAILURE_MODE: {failure_mode}\n"
                "MISTAKES:\n- Returned an incompatible result format.\n"
                "FIXES:\n- Match the required result contract exactly.\n"
            ),
        }
    }


def test_contract_parser_and_compatibility():
    csv = _contract(
        "BigCodeBench/csv",
        "Save data to a CSV file. The function should output with:\n    str: path.",
    )
    csv2 = _contract(
        "BigCodeBench/csv2",
        "Write another CSV. The function should output with:\n    str: file path.",
    )
    url = _contract(
        "BigCodeBench/url",
        "Parse all URLs. The function should output with:\n    list: URLs.",
    )
    assert csv["return_type"] == "str"
    assert "filesystem" in csv["io_families"]
    assert BCBRegionRunner._contract_compatible(csv, csv2)
    assert not BCBRegionRunner._contract_compatible(csv, url)


def test_summary_filters_to_matching_return_contract():
    csv_prompt = (
        "Save data to a CSV file. The function should output with:\n"
        "    str: file path."
    )
    url_prompt = (
        "Parse every URL in text. The function should output with:\n"
        "    list: URLs."
    )
    fake_runner = SimpleNamespace(
        mem=SimpleNamespace(
            _mem_cache={
                "csv_failure": _failure("BigCodeBench/csv", csv_prompt),
                "url_failure": _failure("BigCodeBench/url", url_prompt),
            }
        ),
        _extract_bcb_task_contract=BCBRegionRunner._extract_bcb_task_contract,
        _contract_compatible=BCBRegionRunner._contract_compatible,
    )
    region = SimpleNamespace(member_ids=["csv_failure", "url_failure"])
    query = _contract("BigCodeBench/query", csv_prompt)
    summary, matched, total = BCBRegionRunner._build_contract_filtered_summary(
        fake_runner, region, query
    )
    assert total == 2
    assert matched == 1
    assert "Common failure patterns" in summary



def test_real_injection_path_uses_prompt_when_task_has_only_id():
    """Regression: BCB passes task metadata without the full prompt here."""
    csv_prompt = (
        "Save data to a CSV file. The function should output with:\n"
        "    str: file path."
    )
    url_prompt = (
        "Parse every URL in text. The function should output with:\n"
        "    list: URLs."
    )
    region = SimpleNamespace(member_ids=["csv_failure", "url_failure"])
    fake_runner = SimpleNamespace(
        mem=SimpleNamespace(
            region_manager=SimpleNamespace(regions=[region]),
            _mem_cache={
                "csv_failure": _failure("BigCodeBench/csv", csv_prompt),
                "url_failure": _failure("BigCodeBench/url", url_prompt),
            },
        ),
        _failure_summary_contract_filter=True,
        _failure_summary_lib_filter=False,
        _extract_bcb_task_contract=BCBRegionRunner._extract_bcb_task_contract,
        _build_contract_filtered_summary=(
            lambda region, task_contract: BCBRegionRunner._build_contract_filtered_summary(
                fake_runner, region, task_contract
            )
        ),
        _contract_compatible=BCBRegionRunner._contract_compatible,
    )
    failures = [{"memory_id": "url_failure", "content": "raw"}]
    replaced, dropped = BCBRegionRunner._replace_bcb_failure_with_summary(
        fake_runner,
        failures,
        task={"task_id": "BigCodeBench/query"},
        prompt=csv_prompt,
    )
    assert (replaced, dropped) == (1, 0)
    assert "contract-matched" in failures[0]["content"]


def test_no_compatible_failure_skips_slot():
    url_prompt = (
        "Parse every URL in text. The function should output with:\n"
        "    list: URLs."
    )
    dict_prompt = (
        "Normalize values. The function should output with:\n"
        "    dict: normalized data."
    )
    fake_runner = SimpleNamespace(
        mem=SimpleNamespace(
            _mem_cache={"url_failure": _failure("BigCodeBench/url", url_prompt)}
        ),
        _extract_bcb_task_contract=BCBRegionRunner._extract_bcb_task_contract,
        _contract_compatible=BCBRegionRunner._contract_compatible,
    )
    region = SimpleNamespace(member_ids=["url_failure"])
    summary, matched, total = BCBRegionRunner._build_contract_filtered_summary(
        fake_runner, region, _contract("BigCodeBench/query", dict_prompt)
    )
    assert total == 1
    assert matched == 0
    assert summary == ""


def _conditional_fake_runner():
    return SimpleNamespace(
        mem=SimpleNamespace(region_manager=SimpleNamespace(regions=[object()])),
        retrieve_k=10,
        _failure_summary_n_slots=1,
        _failure_summary_fmmatch=False,
        _failure_summary_replace=False,
        _failure_summary_lib_filter=False,
        _failure_summary_contract_filter=False,
        _failure_summary_force_recall=False,
        _fmmatch_backfill=False,
        _failure_inject_log_counter=0,
        _get_outcome=lambda m: (m.get("metadata") or {}).get("outcome", ""),
    )


def test_failure_summary_abstains_without_selected_failure():
    selected = [
        {"memory_id": f"success_{i}", "metadata": {"outcome": "success"}}
        for i in range(5)
    ]
    result = BCBRegionRunner._inject_failure_summary(
        _conditional_fake_runner(), selected, "query"
    )
    assert result == selected
    assert [m["memory_id"] for m in result] == [m["memory_id"] for m in selected]


def test_failure_summary_preserves_selected_ids_and_order():
    selected = [
        {"memory_id": "s0", "metadata": {"outcome": "success"}},
        {"memory_id": "f0", "metadata": {"outcome": "failure"}},
        {"memory_id": "s1", "metadata": {"outcome": "success"}},
        {"memory_id": "f1", "metadata": {"outcome": "failure"}},
        {"memory_id": "s2", "metadata": {"outcome": "success"}},
    ]
    result = BCBRegionRunner._inject_failure_summary(
        _conditional_fake_runner(), selected, "query"
    )
    assert len(result) == 5
    assert [m["memory_id"] for m in result] == ["s0", "f0", "s1", "f1", "s2"]


def test_force_recall_keeps_fixed_five_slots():
    selected = [
        {"memory_id": f"s{i}", "metadata": {"outcome": "success"}}
        for i in range(5)
    ]
    recalled = {"memory_id": "f0", "metadata": {"outcome": "failure"}}
    runner = _conditional_fake_runner()
    runner._failure_summary_force_recall = True
    runner._retrieve_failure_only_bcb = lambda *_args, **_kwargs: [recalled]
    result = BCBRegionRunner._inject_failure_summary(runner, selected, "query")
    assert len(result) == 5
    assert [m["memory_id"] for m in result] == ["s0", "s1", "s2", "s3", "f0"]


if __name__ == "__main__":
    test_contract_parser_and_compatibility()
    test_summary_filters_to_matching_return_contract()
    test_real_injection_path_uses_prompt_when_task_has_only_id()
    test_no_compatible_failure_skips_slot()
    test_failure_summary_abstains_without_selected_failure()
    test_failure_summary_preserves_selected_ids_and_order()
    test_force_recall_keeps_fixed_five_slots()
    print("contract-aware failure-summary tests: PASS")
