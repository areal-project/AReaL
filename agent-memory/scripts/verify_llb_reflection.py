"""Verify the failure-signal -> reflection pipeline end to end.

Runs in the Singularity container. Makes a few real LLM calls to the configured
MatrixLLM endpoint. No MariaDB / no MemoryService cube needed.

It builds synthetic failed LLB sessions (protocol violation / step limit / wrong
answer), runs the runner's _session_failure_reason() to extract evidence, then
calls AdjustmentUpdater._generate_reflection() with that evidence + benchmark
tag, and prints the resulting reflection. Success criterion: the reflection
names a real failure (protocol/format/answer) instead of "None (Correct)".
"""
import sys
from pathlib import Path

PROJECT = Path("/storage/openpsi/users/yl/agent-memory/MemRL")
sys.path.insert(0, str(PROJECT))

# LLB path + py310 shims are set up by importing the runner module
import memrl.run.llb_rl_runner as R  # noqa: E402  (also fixes sys.path for src.*)
from src.typings import (  # noqa: E402
    Session,
    SampleStatus,
    TaskName,
)
from src.typings.session import (  # noqa: E402
    SessionEvaluationRecord,
    SessionEvaluationOutcome,
)

from memrl.configs.config import MempConfig  # noqa: E402
from memrl.providers.llm import OpenAILLM  # noqa: E402
from memrl.service.updater import AdjustmentUpdater, AdjustmentConfig  # noqa: E402
from memrl.service.strategies import (  # noqa: E402
    BuildStrategy,
    RetrieveStrategy,
    UpdateStrategy,
    StrategyConfiguration,
)

SEP = "=" * 80


def section(t):
    print("\n" + SEP + "\n" + t + "\n" + SEP)


# ---- config + llm ----
cfg = MempConfig.from_yaml(str(PROJECT / "configs/rl_llb_db_config.local.yaml"))
llm = OpenAILLM(
    api_key=cfg.llm.api_key,
    base_url=cfg.llm.base_url,
    model=cfg.llm.model,
    temperature=cfg.llm.temperature,
)
print("llm model:", cfg.llm.model)

strat = StrategyConfiguration(
    build=BuildStrategy(cfg.memory.build_strategy),
    retrieve=RetrieveStrategy(cfg.memory.retrieve_strategy),
    update=UpdateStrategy(cfg.memory.update_strategy),
)

# AdjustmentUpdater needs: mos, user_id, strategies, llm, ...
# We won't call MemOS; only _generate_reflection (LLM-only). Pass mos=None and
# a dummy default_cube_id; prepare_update_op is not invoked here.
updater = AdjustmentUpdater(
    mos=None,
    num_workers=1,
    user_id=cfg.memory.user_id,
    strategies=strat,
    llm=llm,
    default_cube_id="dummy",
    memory_confidence=cfg.memory.memory_confidence,
    adjustment_config=AdjustmentConfig(),
)


# ---- reuse the runner's extractor without constructing a full runner ----
# _session_failure_reason only touches session fields, so bind it unbound.
extract_reason = R.LLBRunner._session_failure_reason.__get__(object())


def make_session(status, outcome, finish_reason=None, detail=None):
    s = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    s.sample_status = status
    s.evaluation_record = SessionEvaluationRecord(
        outcome=outcome, detail_dict=detail
    )
    if finish_reason is not None:
        s.finish_reason = finish_reason
    return s


TASK = ("What are the countries and the range of years of experience "
        "(max - min) for groups where that range is greater than 5? "
        "Return country and range, ordered by country.")

TRAJ = (
    "user: You are a DB agent. Use 'Action: Operation' to run SQL and "
    "'Action: Answer' to submit.\n"
    "assistant: SELECT country, MAX(years)-MIN(years) AS range FROM emp "
    "GROUP BY country HAVING range > 5;\n"
    "user: [(‘US’, 7), (‘UK’, 9)]\n"
    "assistant: The answer is [('US', 7), ('UK', 9)]\n"
)

cases = [
    ("protocol violation",
     make_session(SampleStatus.AGENT_VALIDATION_FAILED,
                  SessionEvaluationOutcome.INCORRECT,
                  finish_reason="Response did not contain a valid 'Action:' directive.")),
    ("step limit",
     make_session(SampleStatus.TASK_LIMIT_REACHED,
                  SessionEvaluationOutcome.INCORRECT)),
    ("wrong answer (status completed)",
     make_session(SampleStatus.COMPLETED,
                  SessionEvaluationOutcome.INCORRECT,
                  detail={"score": 0.0})),
]

for name, sess in cases:
    section(f"CASE: {name}")
    reason = extract_reason(sess)
    print("extracted failure_reason:\n ", reason or "(empty)")
    print("\n--- reflection (source_benchmark=llb_db) ---")
    refl = updater._generate_reflection(
        TASK, TRAJ, eval_error=reason, source_benchmark="llb_db"
    )
    print(refl)
    low = refl.lower()
    bad = ("failure_mode: none" in low
           or "no mistakes" in low
           or "correct query and output" in low
           or "none detected" in low)
    print("\n>>> BAD (claims correct / no mistakes)?", bad)

section("CONTROL: old path (no benchmark tag, no evidence) — should still be code-gen style")
refl_old = updater._generate_reflection(TASK, TRAJ, eval_error="", source_benchmark="")
print(refl_old[:600])
