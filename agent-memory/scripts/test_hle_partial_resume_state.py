#!/usr/bin/env python3
"""Regression test for ID-aware partial-batch HLE checkpoint state."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import types

# Keep the focused checkpoint test independent of optional TensorBoard.
if "torch.utils.tensorboard" not in sys.modules:
    tb = types.ModuleType("torch.utils.tensorboard")
    class _SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def flush(self): pass
    tb.SummaryWriter = _SummaryWriter
    sys.modules["torch.utils.tensorboard"] = tb

# Stub heavy provider/service modules; this test only exercises pure resume helpers.
providers = types.ModuleType("memrl.providers.llm")
providers.OpenAILLM = type("OpenAILLM", (), {})
sys.modules["memrl.providers.llm"] = providers
service = types.ModuleType("memrl.service.memory_service")
service.MemoryService = type("MemoryService", (), {})
sys.modules["memrl.service.memory_service"] = service

# Import through the package so relative imports resolve.
from memrl.run.hle_runner import HLERunner



def _runner(state_path: Path):
    runner = object.__new__(HLERunner)
    runner._cum_state_path = state_path
    runner.train_cumulative_correct_map = {}
    runner.valid_cumulative_correct_map = {}
    runner.holdout_categories = None
    runner._resume_section_start = 0
    runner._resume_batch_start = 0
    runner._resume_batch_all_recs = []
    runner._resume_pending_task_ids = set()
    runner._terminal_incorrect_task_ids = set()
    return runner


def test_id_aware_partial_batch_state_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "cum_state.json"
        writer = _runner(state_path)
        writer._terminal_incorrect_task_ids = {"q_terminal"}
        done = [
            {"id": "q1", "question": "one", "correct": True},
            {"id": "q2", "question": "two", "correct": False},
        ]
        writer._save_cum_state(
            next_section=3,
            next_batch=12,
            batch_all_recs=writer._slim_batch_records(done),
            pending_task_ids=["q3", "q4"],
        )
        raw = json.loads(state_path.read_text())
        assert raw["next_section"] == 3
        assert raw["next_batch"] == 12
        assert raw["pending_task_ids"] == ["q3", "q4"]
        assert raw["batch_all_recs"][0]["id"] == "q1"
        assert raw["terminal_incorrect_task_ids"] == ["q_terminal"]

        resumed = _runner(state_path)
        resumed._load_cum_state()
        restored = resumed._normalize_resume_batch_records(resumed._resume_batch_all_recs)
        completed = {str(r["id"]) for r in restored if r.get("id") is not None}
        assert completed == {"q1", "q2"}
        assert resumed._resume_pending_task_ids == {"q3", "q4"}
        assert resumed._terminal_incorrect_task_ids == {"q_terminal"}

        # The next run skips persisted successes but keeps pending and future IDs.
        batch_ids = ["q1", "q2", "q3", "q4", "q5"]
        scheduled = [qid for qid in batch_ids if qid not in completed]
        assert scheduled == ["q3", "q4", "q5"]


def test_legacy_tuple_records_remain_readable():
    restored = HLERunner._normalize_resume_batch_records([("legacy question", True)])
    assert restored == [{"id": None, "question": "legacy question", "correct": True}]


def test_compact_pending_schedule_starts_at_saved_cursor():
    import pandas as pd

    runner = object.__new__(HLERunner)
    runner.batch_size = 4
    runner._resume_pending_task_ids = {"q1", "q6", "q9"}
    df = pd.DataFrame([{"id": f"q{i}", "question": str(i)} for i in range(12)])
    batches = [list(range(i, min(i + 4, len(df)))) for i in range(0, len(df), 4)]
    completed = {"q0", "q2", "q3", "q4", "q5", "q7"}

    schedule = runner._build_train_batch_schedule(
        df, batches, start_batch_idx=2, completed_task_ids=completed,
        legacy_resume_without_ids=False, sec_idx=3,
    )
    # All scattered pending IDs are grouped once; original execution resumes at
    # batch 2 instead of scanning batches 0 and 1 as tiny executors.
    assert schedule[0] == (None, [1, 6, 9])
    assert schedule[1:] == [(2, [8, 9, 10, 11])]


def test_legacy_schedule_preserves_old_cursor_behavior():
    import pandas as pd

    runner = object.__new__(HLERunner)
    runner.batch_size = 2
    runner._resume_pending_task_ids = set()
    df = pd.DataFrame([{"id": f"q{i}"} for i in range(6)])
    batches = [[0, 1], [2, 3], [4, 5]]
    schedule = runner._build_train_batch_schedule(
        df, batches, start_batch_idx=2, completed_task_ids=set(),
        legacy_resume_without_ids=True, sec_idx=1,
    )
    assert schedule == [(0, [0, 1]), (1, [2, 3]), (2, [4, 5])]
def test_pending_timeout_becomes_terminal_incorrect_without_waiting():
    import os
    import pandas as pd
    import time

    runner = object.__new__(HLERunner)
    runner.batch_size = 2
    runner._terminal_incorrect_task_ids = set()
    runner._infrastructure_deferred_task_ids = set()
    def slow_eval(row):
        time.sleep(0.3)
        return {"id": row["id"], "question": row["question"], "correct": True, "trajectory": "late"}
    runner._evaluate_row = slow_eval
    old = os.environ.get("MEMRL_RESUME_PENDING_TIMEOUT_S")
    os.environ["MEMRL_RESUME_PENDING_TIMEOUT_S"] = "0.05"
    started = time.monotonic()
    try:
        rows = [pd.Series({"id":"q1","question":"one"}), pd.Series({"id":"q2","question":"two"})]
        results, deferred = runner._evaluate_resume_pending_batch(rows, sec_idx=3)
    finally:
        if old is None: os.environ.pop("MEMRL_RESUME_PENDING_TIMEOUT_S", None)
        else: os.environ["MEMRL_RESUME_PENDING_TIMEOUT_S"] = old
    elapsed = time.monotonic() - started
    assert elapsed < 0.2, elapsed
    assert len(results) == 2 and deferred == []
    assert all(r["correct"] is False and r["terminal_incorrect"] for r in results)
    assert all(r["trajectory"] == "" for r in results)
def test_completed_api_overload_is_deferred_not_incorrect():
    import pandas as pd
    runner = object.__new__(HLERunner)
    runner.batch_size = 1
    runner._terminal_incorrect_task_ids = set()
    runner._infrastructure_deferred_task_ids = set()
    runner._evaluate_row = lambda row: {
        "id": row["id"], "question": row["question"], "correct": False,
        "trajectory": "", "gen_error": "Error code: 500 stream header timeout adapter_api_error",
    }
    rows=[pd.Series({"id":"infra1","question":"q"})]
    scored, deferred = runner._evaluate_resume_pending_batch(rows, sec_idx=3)
    assert scored == [] and len(deferred) == 1
    assert runner._terminal_incorrect_task_ids == set()
    assert runner._infrastructure_deferred_task_ids == {"infra1"}


def test_pending_checkpoint_keeps_cursor_and_terminal_ids():
    import json
    import tempfile
    from pathlib import Path
    runner = object.__new__(HLERunner)
    runner.memory_service = type('MS', (), {'save_checkpoint_snapshot': lambda self, ck, ckpt_id: {'checkpoint_id': ckpt_id}})()
    runner.ck_dir = Path(tempfile.mkdtemp())
    (runner.ck_dir / 'snapshot' / '3_pending' / 'cube').mkdir(parents=True)
    runner._cum_state_path = runner.ck_dir / 'local_cache' / 'cum_state.json'
    runner._cum_state_path.parent.mkdir(parents=True)
    runner.train_cumulative_correct_map = {}
    runner.valid_cumulative_correct_map = {}
    runner.holdout_categories = None
    runner._resume_pending_task_ids = {'q2', 'q3'}
    runner._terminal_incorrect_task_ids = {'q2'}
    runner._infrastructure_deferred_task_ids = set()
    recs = [
        {'id':'q1','question':'one','correct':True},
        {'id':'q2','question':'two','correct':False},
    ]
    runner._save_pending_recovery_checkpoint(3, 50, recs, {'q1','q2'})
    raw=json.loads((runner.ck_dir/'snapshot'/'3_pending'/'local_cache'/'cum_state.json').read_text())
    assert raw['next_section']==3 and raw['next_batch']==50
    assert raw['pending_task_ids']==['q3']
    assert raw['terminal_incorrect_task_ids']==['q2']
    assert len(raw['batch_all_recs'])==2
def test_infrastructure_classifier_examples():
    assert HLERunner._is_infrastructure_error_text('Error code: 500 stream header timeout adapter_api_error')
    assert HLERunner._is_infrastructure_error_text('429 RESOURCE_EXHAUSTED')
    assert HLERunner._is_infrastructure_error_text('Connection error')
    assert not HLERunner._is_infrastructure_error_text('empty_response')
    assert not HLERunner._is_infrastructure_error_text('model returned invalid answer')
def test_stable_id_dedup_keeps_first_result():
    records=[
        {'id':'q1','question':'one','correct':False},
        {'id':'q1','question':'one duplicate','correct':True},
        {'id':'q2','question':'two','correct':True},
    ]
    out=HLERunner._stable_dedup_records(records)
    assert len(out)==2
    assert out[0]['id']=='q1' and out[0]['correct'] is False
    assert out[1]['id']=='q2'
def test_reconcile_removes_scored_ids_from_pending_and_deferred():
    runner = object.__new__(HLERunner)
    runner._resume_pending_task_ids = {'q1','q2','q3'}
    runner._infrastructure_deferred_task_ids = {'q2','q3','q4'}
    runner._resume_batch_all_recs = []
    runner._reconcile_resume_task_sets({'q2','q4'})
    assert runner._resume_pending_task_ids == {'q1','q3'}
    assert runner._infrastructure_deferred_task_ids == {'q3'}



if __name__ == "__main__":
    test_id_aware_partial_batch_state_roundtrip()
    test_legacy_tuple_records_remain_readable()
    test_compact_pending_schedule_starts_at_saved_cursor()
    test_legacy_schedule_preserves_old_cursor_behavior()
    test_pending_timeout_becomes_terminal_incorrect_without_waiting()
    test_completed_api_overload_is_deferred_not_incorrect()
    test_pending_checkpoint_keeps_cursor_and_terminal_ids()
    test_infrastructure_classifier_examples()
    test_stable_id_dedup_keeps_first_result()
    test_reconcile_removes_scored_ids_from_pending_and_deferred()
    print("OK: task-ID dedup and pending/deferred reconciliation are safe")
