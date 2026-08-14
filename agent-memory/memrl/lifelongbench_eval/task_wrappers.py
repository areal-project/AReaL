"""LLB task wrappers and utilities."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Iterable, Tuple

# Setup LLB path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
LLB_ROOT = _project_root / "3rdparty" / "LifelongAgentBench"

if not LLB_ROOT.exists():
    raise RuntimeError(f"LLB directory not found: {LLB_ROOT}")

# Patch LLB containers to use Apptainer when available (must run before LLB imports)
from memrl.apptainer.patch import patch_llb_containers
patch_llb_containers()

# Python 3.10 compatibility
try:
    import enum as _enum

    if not hasattr(_enum, "StrEnum"):

        class _StrEnum(str, _enum.Enum):
            pass

        _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
    import typing as _typing

    if not hasattr(_typing, "reveal_type"):

        def _noop_reveal_type(x):
            return x

        _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]
    if not hasattr(_typing, "Self"):
        _typing.Self = object  # type: ignore[attr-defined]
except Exception:
    pass

if str(LLB_ROOT) not in sys.path:
    sys.path.insert(0, str(LLB_ROOT))

from src.typings import (  # type: ignore
    Session,
    SampleStatus,
    SessionMetricCalculationPartial,
    TaskName,
    SessionEvaluationOutcome,
)
from src.factories.chat_history_item.offline.construct import (  # type: ignore
    construct_offline,
)
from src.factories.chat_history_item import ChatHistoryItemFactory  # type: ignore
from src.agents.instance.language_model_agent import (  # type: ignore
    LanguageModelAgent,
)


def ensure_standard_prompts() -> str:
    """Ensure standard prompt JSON files are generated."""
    standard_dir = LLB_ROOT / "chat_history_items" / "standard"
    if not standard_dir.exists() or not (standard_dir / "db_bench.json").is_file():
        standard_dir.mkdir(parents=True, exist_ok=True)
        construct_offline(
            str(standard_dir),
            "\n\nNow, I will give you the question that you need to solve.",
        )
    return str(standard_dir)


def build_task(
    task: str,
    data_file_path: str,
    *,
    max_round: int,
    os_timeout: int = 20,
    kg_sparql_url: str | None = None,
    kg_ontology_dir: str | None = None,
    kg_offline_fallback: bool = False,
) -> Tuple[Any | None, TaskName]:
    """Build LLB task object."""
    standard_dir = ensure_standard_prompts()

    if task == "db" or task == "db_bench":
        try:
            from src.tasks.instance.db_bench.task import DBBench  # type: ignore
        except ModuleNotFoundError as e:
            _name = getattr(e, "name", "")
            if _name in ("docker", "docker.client"):
                raise RuntimeError("Missing docker SDK: pip install docker") from e
            if _name in ("mysql", "mysql.connector"):
                raise RuntimeError(
                    "Missing MySQL connector: pip install mysql-connector-python"
                ) from e
            raise RuntimeError("Failed to import LLB task module") from e

        tname = TaskName.DB_BENCH
        factory = ChatHistoryItemFactory(
            chat_history_item_dict_path=os.path.join(standard_dir, "db_bench.json")
        )
        task_obj = DBBench(
            task_name=tname,
            chat_history_item_factory=factory,
            data_file_path=data_file_path,
            max_round=max_round,
        )
        return task_obj, tname

    elif task == "os" or task == "os_interaction":
        try:
            from src.tasks.instance.os_interaction.task import OSInteraction  # type: ignore
        except ModuleNotFoundError as e:
            _name = getattr(e, "name", "")
            if _name in ("docker", "docker.client"):
                raise RuntimeError("Missing docker SDK: pip install docker") from e
            raise RuntimeError("Failed to import LLB task module") from e

        tname = TaskName.OS_INTERACTION
        factory = ChatHistoryItemFactory(
            chat_history_item_dict_path=os.path.join(
                standard_dir, "os_interaction.json"
            )
        )
        task_obj = OSInteraction(
            task_name=tname,
            chat_history_item_factory=factory,
            data_file_path=data_file_path,
            max_round=max_round,
            command_execution_timeout=os_timeout,
        )
        return task_obj, tname

    elif task == "kg" or task == "knowledge_graph":
        tname = TaskName.KNOWLEDGE_GRAPH
        if kg_offline_fallback:
            return None, tname

        if not kg_sparql_url or not kg_ontology_dir:
            raise ValueError("KG task requires --sparql-url and --ontology-dir")

        factory = ChatHistoryItemFactory(
            chat_history_item_dict_path=os.path.join(
                standard_dir, "knowledge_graph.json"
            )
        )
        try:
            from src.tasks.instance.knowledge_graph.task import KnowledgeGraph  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError("Failed to import KnowledgeGraph task") from e

        task_obj = KnowledgeGraph(
            task_name=tname,
            chat_history_item_factory=factory,
            sparql_url=kg_sparql_url,
            ontology_dir_path=kg_ontology_dir,
            data_file_path=data_file_path,
            max_round=max_round,
        )
        return task_obj, tname

    else:
        raise ValueError(f"Unsupported task type: {task}")


def sorted_sample_indices(entries_json_path: str, limit: int | None) -> list[str]:
    """Get sorted sample indices from dataset."""
    with open(entries_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = sorted(list(data.keys()), key=lambda x: str(x))
    if limit is not None:
        keys = keys[: max(0, int(limit))]
    return keys


def run_sessions(
    task_obj: Any,
    task_name: TaskName,
    agent: LanguageModelAgent,
    sample_indices: Iterable[str],
) -> Tuple[list[Session], dict]:
    """Run sessions for given sample indices."""
    sessions: list[Session] = []
    for sid in sample_indices:
        session = Session(task_name=task_name, sample_index=sid)
        task_obj.reset(session)
        while session.sample_status == SampleStatus.RUNNING:
            agent.inference(session)
            task_obj.interact(session)
        task_obj.complete(session)
        sessions.append(session)

    # Calculate metrics
    partials = [
        SessionMetricCalculationPartial(
            sample_index=s.sample_index,
            evaluation_record=s.evaluation_record,
            sample_status=s.sample_status,
        )
        for s in sessions
    ]
    metrics = task_obj.calculate_metric(partials)
    return sessions, metrics


def save_outputs(out_dir: str, sessions: list[Session], metrics: dict) -> None:
    """Save sessions and metrics to output directory."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "runs.json"), "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in sessions], f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
