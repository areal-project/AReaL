import sys

from loguru import logger
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType

from .task_runner_socket import (
    _build_cli_parser,
    _configure_logging,
    _load_selected_task,
    _log_simulation_summary,
    _run_task_impl,
)


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: str | None = None,
    llm_args_agent: dict | None = None,
    llm_user: str | None = None,
    llm_args_user: dict | None = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: int | None = None,
    enforce_communication_protocol: bool = False,
) -> SimulationRun:
    return _run_task_impl(
        domain,
        task,
        agent,
        user,
        llm_agent,
        llm_args_agent,
        llm_user,
        llm_args_user,
        max_steps,
        max_errors,
        evaluation_type,
        seed,
        enforce_communication_protocol,
        use_socket_server=False,
        initialize_environment=False,
        save_simulation=False,
    )[0]


def main() -> int:
    args = _build_cli_parser("Test OpenClaw agent with TAU² tasks", 20).parse_args()
    _configure_logging(args.log_level)
    task = _load_selected_task(args.domain, args.task_id)
    if task is None:
        return 1
    try:
        simulation = run_task(
            args.domain,
            task,
            args.agent,
            args.user,
            args.llm_agent,
            None,
            args.llm_user,
            None,
            args.max_steps,
            5,
            EvaluationType.ALL,
            args.seed,
            False,
        )
    except Exception:
        logger.exception("Error running task")
        return 1
    _log_simulation_summary(simulation, debug=args.log_level == "DEBUG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
