from loguru import logger
from tau2.data_model.simulation import (
    DBCheck,
    EnvAssertionCheck,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.registry import registry

_STOP_REASONS = {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}


def _note_reward(note: str) -> RewardInfo:
    return RewardInfo(reward=1.0, reward_basis=None, info={"note": note})


def _initial_state_kwargs(task: Task) -> dict:
    state = task.initial_state
    return {
        "initialization_data": getattr(state, "initialization_data", None),
        "initialization_actions": getattr(state, "initialization_actions", None),
        "message_history": getattr(state, "message_history", None) or [],
    }


def _merge_reward_breakdowns(*infos: RewardInfo | None) -> dict:
    return {
        key: value
        for info in infos
        if info and info.reward_breakdown
        for key, value in info.reward_breakdown.items()
    }


class OpenClawEnvironmentEvaluator:
    @classmethod
    def calculate_reward(
        cls,
        environment: Environment,
        task: Task,
        full_trajectory: list,
        solo_mode: bool = False,
    ) -> RewardInfo:
        _ = full_trajectory, solo_mode
        criteria = task.evaluation_criteria
        if criteria is None:
            return _note_reward("No evaluation criteria")
        if not criteria.actions and not criteria.env_assertions:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                reward_basis=criteria.reward_basis,
                info={"note": "No expected actions or env assertions"},
            )
        gold_environment = registry.get_env_constructor(environment.domain_name)()
        gold_environment.set_state(**_initial_state_kwargs(task))
        for action in criteria.actions or []:
            try:
                gold_environment.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception as exc:
                logger.warning(
                    "Error in golden action {}({}): {}",
                    action.name,
                    action.arguments,
                    exc,
                )
        db_reward = float(
            gold_environment.get_db_hash() == environment.get_db_hash()
            and gold_environment.get_user_db_hash() == environment.get_user_db_hash()
        )
        env_checks: list[EnvAssertionCheck] = []
        env_reward = 1.0
        for assertion in criteria.env_assertions or []:
            met = environment.run_env_assertion(assertion, raise_assertion_error=False)
            check = EnvAssertionCheck(
                env_assertion=assertion, met=met, reward=float(met)
            )
            env_checks.append(check)
            env_reward *= check.reward
        reward = 1.0
        reward_breakdown = {}
        if RewardType.DB in criteria.reward_basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_reward
            reward *= env_reward
        return RewardInfo(
            reward=reward,
            db_check=DBCheck(db_match=bool(db_reward), db_reward=db_reward),
            env_assertions=env_checks,
            reward_basis=criteria.reward_basis,
            reward_breakdown=reward_breakdown,
        )


def evaluate_simulation_with_environment(
    simulation: SimulationRun,
    task: Task,
    environment: Environment,
    evaluation_type: EvaluationType,
    solo_mode: bool = False,
) -> RewardInfo:
    if simulation.termination_reason not in _STOP_REASONS:
        return RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={
                "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason.value}"
            },
        )
    criteria = task.evaluation_criteria
    if criteria is None:
        return _note_reward("No evaluation criteria")
    if evaluation_type == EvaluationType.ENV:
        return OpenClawEnvironmentEvaluator.calculate_reward(
            environment, task, simulation.messages, solo_mode
        )
    if evaluation_type == EvaluationType.NL_ASSERTIONS:
        return NLAssertionsEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages
        )
    if evaluation_type == EvaluationType.COMMUNICATE:
        return CommunicateEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages
        )
    if evaluation_type == EvaluationType.ACTION:
        return ActionEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages
        )
    if evaluation_type not in {
        EvaluationType.ALL,
        EvaluationType.ALL_WITH_NL_ASSERTIONS,
    }:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    env_info = OpenClawEnvironmentEvaluator.calculate_reward(
        environment, task, simulation.messages, solo_mode
    )
    action_info = ActionEvaluator.calculate_reward(
        task=task, full_trajectory=simulation.messages
    )
    communicate_info = CommunicateEvaluator.calculate_reward(
        task=task, full_trajectory=simulation.messages
    )
    nl_info = (
        NLAssertionsEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages
        )
        if evaluation_type == EvaluationType.ALL_WITH_NL_ASSERTIONS
        else None
    )
    reward = 1.0
    reward_basis = set(criteria.reward_basis)
    for basis_group, info in (
        ({RewardType.DB, RewardType.ENV_ASSERTION}, env_info),
        ({RewardType.ACTION}, action_info),
        ({RewardType.COMMUNICATE}, communicate_info),
    ):
        if reward_basis & basis_group:
            reward *= info.reward
    if RewardType.NL_ASSERTION in reward_basis:
        if nl_info is None:
            raise ValueError(
                "NL assertions are part of the reward basis, but they are not being evaluated."
            )
        reward *= nl_info.reward
    return RewardInfo(
        reward=reward,
        db_check=env_info.db_check,
        env_assertions=env_info.env_assertions,
        action_checks=action_info.action_checks,
        nl_assertions=nl_info.nl_assertions if nl_info else None,
        communicate_checks=communicate_info.communicate_checks,
        reward_basis=criteria.reward_basis,
        reward_breakdown=_merge_reward_breakdowns(
            env_info, action_info, communicate_info, nl_info
        ),
        info={
            "env": env_info.info,
            "nl": nl_info.info if nl_info else None,
            "communicate": communicate_info.info,
            "action": action_info.info,
        },
    )
