import inspect
import json
import sys
from pathlib import Path

from loguru import logger
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry

from .tau2_env import EnvironmentSocketServer, evaluate_simulation_with_environment


def _initial_state_kwargs(task: Task) -> dict:
    state = task.initial_state
    return {
        "initialization_data": getattr(state, "initialization_data", None),
        "initialization_actions": getattr(state, "initialization_actions", None),
        "message_history": getattr(state, "message_history", None) or [],
    }


def _save_simulation(
    domain: str, task: Task, agent: str, simulation: SimulationRun
) -> None:
    output_file = (
        Path("simulation_results") / f"simulation_{domain}_{task.id}_{agent}.json"
    )
    output_file.parent.mkdir(exist_ok=True)
    try:
        payload = (
            simulation.model_dump()
            if hasattr(simulation, "model_dump")
            else simulation.dict()
            if hasattr(simulation, "dict")
            else simulation.__dict__
        )
        output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Simulation saved to: {}", output_file)
    except Exception as exc:
        logger.error("Failed to save simulation to file: {}", exc)


def _build_agent(
    task: Task,
    agent: str,
    llm_agent: str | None,
    llm_args_agent: dict | None,
    environment: Environment,
    environment_constructor,
    env_server: EnvironmentSocketServer | None,
    server_config: dict | None,
    init_kwargs: dict,
):
    from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent

    try:
        from tau2.gym.gym_agent import GymAgent
    except ImportError:
        GymAgent = None  # noqa: N806
    agent_constructor = registry.get_agent_constructor(agent)
    tools = environment.get_tools()
    policy = environment.get_policy()
    if issubclass(agent_constructor, LLMAgent):
        return (
            agent_constructor(
                tools=tools,
                domain_policy=policy,
                llm=llm_agent,
                llm_args=llm_args_agent,
            ),
            environment,
            False,
        )
    if issubclass(agent_constructor, LLMGTAgent):
        return (
            agent_constructor(
                tools=tools,
                domain_policy=policy,
                llm=llm_agent,
                llm_args=llm_args_agent,
                task=task,
            ),
            environment,
            False,
        )
    if issubclass(agent_constructor, LLMSoloAgent):
        solo_environment = environment_constructor(solo_mode=True)
        if task.initial_state:
            solo_environment.set_state(**init_kwargs)
        if env_server:
            env_server.environment = solo_environment
        user_tools = (
            solo_environment.get_user_tools() if solo_environment.user_tools else []
        )
        return (
            agent_constructor(
                tools=solo_environment.get_tools() + user_tools,
                domain_policy=solo_environment.get_policy(),
                llm=llm_agent,
                llm_args=llm_args_agent,
                task=task,
            ),
            solo_environment,
            True,
        )
    if GymAgent is not None and issubclass(agent_constructor, GymAgent):
        return agent_constructor(tools=tools, domain_policy=policy), environment, False
    if agent == "openclaw_agent":
        instance = agent_constructor(
            tools=tools,
            domain_policy=policy,
            socket_server_config=server_config if env_server else None,
        )
        logger.info("OpenClaw agent created with socket config: {}", server_config)
        return instance, environment, False
    return agent_constructor(tools=tools, domain_policy=policy), environment, False


def _run_task_impl(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: str | None,
    llm_args_agent: dict | None,
    llm_user: str | None,
    llm_args_user: dict | None,
    max_steps: int,
    max_errors: int,
    evaluation_type: EvaluationType,
    seed: int | None,
    enforce_communication_protocol: bool,
    *,
    use_socket_server: bool,
    initialize_environment: bool,
    save_simulation: bool,
    socket_port: int | None = None,
) -> tuple[SimulationRun, Environment]:
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    logger.info(
        "STARTING SIMULATION: domain={} task={} agent={} user={}",
        domain,
        task.id,
        agent,
        user,
    )
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    init_kwargs = _initial_state_kwargs(task)
    if initialize_environment and task.initial_state:
        environment.set_state(**init_kwargs)
    env_server = server_config = None
    if use_socket_server and agent == "openclaw_agent":
        env_server = EnvironmentSocketServer(
            environment=environment, task=task, host="127.0.0.1", port=socket_port or 0
        )
        env_server.start()
        server_config = env_server.get_client_config()
    try:
        agent_instance, environment, solo_mode = _build_agent(
            task,
            agent,
            llm_agent,
            llm_args_agent,
            environment,
            environment_constructor,
            env_server,
            server_config,
            init_kwargs,
        )
        from tau2.user.user_simulator import DummyUser

        user_constructor = registry.get_user_constructor(user)
        if issubclass(user_constructor, DummyUser):
            from tau2.agent.llm_agent import LLMSoloAgent

            assert isinstance(agent_instance, LLMSoloAgent), (
                "Dummy user can only be used with solo agent"
            )
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        user_instance = user_constructor(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=llm_user,
            llm_args=llm_args_user,
        )
        orch_kwargs = {
            "domain": domain,
            "agent": agent_instance,
            "user": user_instance,
            "environment": environment,
            "task": task,
            "max_steps": max_steps,
            "max_errors": max_errors,
            "seed": seed,
            "solo_mode": solo_mode,
        }
        if (
            "validate_communication"
            in inspect.signature(Orchestrator.__init__).parameters
        ):
            orch_kwargs["validate_communication"] = enforce_communication_protocol
        simulation = Orchestrator(**orch_kwargs).run()
        simulation.reward_info = evaluate_simulation_with_environment(
            simulation=simulation,
            task=task,
            environment=environment,
            evaluation_type=evaluation_type,
            solo_mode=solo_mode,
        )
        if save_simulation:
            _save_simulation(domain, task, agent, simulation)
        logger.info(
            "FINISHED SIMULATION: domain={} task={} agent={} user={} reward={}",
            domain,
            task.id,
            agent_instance.__class__.__name__,
            user_instance.__class__.__name__,
            simulation.reward_info.reward,
        )
        return simulation, environment
    finally:
        if env_server:
            logger.info("Stopping Environment Socket Server...")
            env_server.stop()


def run_task_with_socket_server(
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
    socket_port: int | None = None,
) -> tuple[SimulationRun, Environment]:
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
        use_socket_server=True,
        initialize_environment=True,
        save_simulation=True,
        socket_port=socket_port,
    )


def _build_cli_parser(description: str, max_steps: int):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--domain", type=str, default="airline", choices=["airline", "retail"]
    )
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    for name, default in (
        ("--agent", "openclaw_agent"),
        ("--user", "user_simulator"),
        ("--llm-agent", "anthropic/claude-3-5-sonnet-20241022"),
        ("--llm-user", "openai/gpt-4.1-2025-04-14"),
    ):
        parser.add_argument(name, type=str, default=default)
    return parser


def _configure_logging(log_level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level)


def _load_selected_task(domain: str, task_id: str | None) -> Task | None:
    from tau2.run import load_tasks

    tasks = load_tasks(task_set_name=domain)
    if not tasks:
        logger.error("No tasks found for domain: {}", domain)
        return None
    task = (
        next((item for item in tasks if item.id == task_id), tasks[0])
        if task_id
        else tasks[0]
    )
    if task_id and task.id != task_id:
        logger.error("Task {} not found", task_id)
        return None
    return task


def _log_simulation_summary(
    simulation: SimulationRun,
    *,
    environment: Environment | None = None,
    debug: bool = False,
) -> None:
    logger.info("Task ID: {}", simulation.task_id)
    logger.info("Steps taken: {}", len(simulation.messages))
    logger.info("Termination reason: {}", simulation.termination_reason)
    logger.info("Duration: {:.2f}s", simulation.duration)
    if simulation.reward_info:
        logger.info("Reward: {}", simulation.reward_info.reward)
        if simulation.reward_info.db_check:
            logger.info("DB Match: {}", simulation.reward_info.db_check.db_match)
        if simulation.reward_info.action_checks:
            logger.info(
                "Action Checks: {}/{} matched",
                sum(
                    check.action_match for check in simulation.reward_info.action_checks
                ),
                len(simulation.reward_info.action_checks),
            )
        if simulation.reward_info.env_assertions:
            logger.info(
                "Environment Assertions: {}/{} met",
                sum(check.met for check in simulation.reward_info.env_assertions),
                len(simulation.reward_info.env_assertions),
            )
    if simulation.agent_cost:
        logger.info("Agent Cost: ${:.4f}", simulation.agent_cost)
    if simulation.user_cost:
        logger.info("User Cost: ${:.4f}", simulation.user_cost)
    if environment is not None:
        logger.info("Environment DB Hash: {}", environment.get_db_hash())
    if debug:
        for idx, turn in enumerate(simulation.messages, 1):
            logger.debug(
                "Turn {}: role={} content={}",
                idx,
                turn.role,
                (turn.content or "")[:200],
            )


def main() -> int:
    args = _build_cli_parser(
        "Run TAU² task with OpenClaw agent using Socket Server", 100
    ).parse_args()
    _configure_logging(args.log_level)
    logger.info("Loading tasks for domain: {}", args.domain)
    task = _load_selected_task(args.domain, args.task_id)
    if task is None:
        return 1
    logger.info("Selected task: {}", task.id)
    try:
        simulation, environment = run_task_with_socket_server(
            args.domain,
            task,
            args.agent,
            args.user,
            args.llm_agent,
            None,
            args.llm_user,
            None,
            args.max_steps,
            10,
            EvaluationType.ALL,
            args.seed,
            False,
        )
    except Exception:
        logger.exception("Error running task")
        return 1
    _log_simulation_summary(
        simulation, environment=environment, debug=args.log_level == "DEBUG"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
