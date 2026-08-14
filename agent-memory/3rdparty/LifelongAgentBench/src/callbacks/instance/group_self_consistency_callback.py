from typing import Any, Mapping, Optional, Sequence, Final
from pydantic import BaseModel
import os
import json
from enum import StrEnum
import re
import inspect
import sqlglot
import datetime

from src.callbacks.callback import Callback, CallbackArguments
from src.language_models import LanguageModel
from src.typings import (
    Session,
    TaskName,
    ChatHistory,
    SampleIndex,
    Role,
    ChatHistoryItem,
    SessionEvaluationOutcome,
    LanguageModelOutOfMemoryException,
)
from src.tasks.task import AgentAction
from src.tasks.instance.db_bench.task import DBBench
from src.tasks.instance.os_interaction.task import OSInteraction
from src.factories.chat_history_item.offline.task_requirement import (
    TASK_REQUIREMENT_DICT,
)
from src.utils import SafeLogger


# region Definition of SessionWrapper and ChatHistoryInfo
class SessionWrapper(BaseModel):
    session: Session
    priority: int
    experience_question: str
    experience_solution: str
    created_time: str

    def __lt__(self, other: "SessionWrapper") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        self_time = datetime.datetime.strptime(
            self.created_time, "%Y-%m-%d %H:%M:%S"
        ).timestamp()
        other_time = datetime.datetime.strptime(
            other.created_time, "%Y-%m-%d %H:%M:%S"
        ).timestamp()
        return self_time < other_time


class ChatHistoryInfo(BaseModel):
    chat_history: ChatHistory
    sample_index_list: Sequence[SampleIndex]


# endregion
# region Definition of SelfConsistencyEntry
# region Definition of RelevanceJudgement and RelevanceInfo
class RelevanceJudgement(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class RelevanceInfo(BaseModel):
    raw_inference: str
    judgement: RelevanceJudgement


# endregion
# region Definition of GroupInfo and InferenceOutcome
class GroupInfo(BaseModel):
    group_index: int
    experience_sample_index_list: Optional[Sequence[SampleIndex]]
    raw_inference: str
    extracted_action: Optional[str]


class SelfConsistencyInferenceOutcome(BaseModel):
    group_info_list: Sequence[GroupInfo]
    action_to_sample_index_list_dict: Optional[Mapping[str, Sequence[SampleIndex]]]
    selected_action: Optional[str]  # How to select the action from candidate actions
    selected_reason: str


# endregion
class SelfConsistencyEntry(BaseModel):
    current_session_index: SampleIndex
    relevance_info_dict: Mapping[SampleIndex, RelevanceInfo]
    inference_outcome_list: list[SelfConsistencyInferenceOutcome]
    current_session_to_priority_dict: Optional[Mapping[SampleIndex, int]]


# endregion
# region Definition of SelfConsistencyBatchSizeManager
class SelfConsistencyPhase(StrEnum):
    RELEVANCE_JUDGEMENT = "relevance_judgement"
    EXPERIENCE_UTILIZATION = "experience_utilization"


class SelfConsistencyBatchSizeManager:
    def __init__(self, batch_size_dict: dict[SelfConsistencyPhase, int]):
        for batch_size in batch_size_dict.values():
            assert batch_size > 0
            # Check if the batch_size is a power of 2
            # https://www.geeksforgeeks.org/python-program-to-find-whether-a-no-is-power-of-two/
            assert (batch_size & (batch_size - 1)) == 0
        self.current_batch_size_dict = batch_size_dict
        self.usage_history: dict[SelfConsistencyPhase, list[int]] = {
            phase: [] for phase in SelfConsistencyPhase
        }
        self.update_history: dict[SelfConsistencyPhase, list[dict[str, int]]] = {
            phase: [] for phase in SelfConsistencyPhase
        }
        self.tolerance: int = 5

    def get(self, phase: SelfConsistencyPhase) -> int:
        return self.current_batch_size_dict[phase]

    def record_usage(self, phase: SelfConsistencyPhase, batch_size: int) -> None:
        self.usage_history[phase].append(batch_size)

    def update(self, phase: SelfConsistencyPhase) -> None:
        if self.current_batch_size_dict[phase] == 1:
            # Current batch size is already 1, cannot be further reduced
            return
        phase_update_history = self.update_history[phase]
        if len(phase_update_history) == 0:
            start_index = 0
        else:
            start_index = phase_update_history[-1]["current_usage_history_length"]
        if start_index > len(self.usage_history[phase]) - self.tolerance:
            # Avoid updating too frequently
            return
        recent_usage_list = self.usage_history[phase][-self.tolerance :]
        if all(
            [
                recent_usage < self.current_batch_size_dict[phase]
                for recent_usage in recent_usage_list
            ]
        ):
            self.current_batch_size_dict[phase] //= 2
            SafeLogger.warning(
                f"[GroupSelfConsistencyCallback] "
                f"Reduce batch size to {self.current_batch_size_dict[phase]} for phase {phase}"
            )
            self.update_history[phase].append(
                {
                    "current_usage_history_length": len(self.usage_history[phase]),
                    "updated_batch_size": self.current_batch_size_dict[phase],
                }
            )

    def load_state(self, state_path: str) -> None:
        state_dict = json.load(open(state_path, "r"))
        self.current_batch_size_dict = {
            SelfConsistencyPhase(phase): batch_size
            for phase, batch_size in state_dict["current_batch_size_dict"].items()
        }
        self.usage_history = {
            SelfConsistencyPhase(phase): usage_history
            for phase, usage_history in state_dict["usage_history"].items()
        }
        self.update_history = {
            SelfConsistencyPhase(phase): update_history
            for phase, update_history in state_dict["update_history"].items()
        }
        self.tolerance = state_dict["tolerance"]

    def dump_state(self, state_path: str) -> None:
        state_dict = {
            "current_batch_size_dict": self.current_batch_size_dict,
            "usage_history": self.usage_history,
            "update_history": self.update_history,
            "tolerance": self.tolerance,
        }
        json.dump(state_dict, open(state_path, "w"), indent=2)  # noqa


# endregion
class GroupSelfConsistencyCallback(Callback):
    def __init__(
        self,
        group_count: Optional[int],
        sample_count_per_group: int,
        batch_size_dict: dict[SelfConsistencyPhase | str, int],
        language_model: LanguageModel,
        task_name: TaskName,
        inference_config_dict: Optional[Mapping[str, Any]],
    ):
        super().__init__()
        assert group_count is None or group_count > 0
        self.group_count: Final[Optional[int]] = group_count
        self.sample_count_per_group: Final[int] = sample_count_per_group
        self.batch_size_manager: Final[SelfConsistencyBatchSizeManager] = (
            SelfConsistencyBatchSizeManager(
                {
                    SelfConsistencyPhase(phase): batch_size
                    for phase, batch_size in batch_size_dict.items()
                }
            )
        )
        self.language_model: Final[LanguageModel] = language_model
        self.task_name: Final[TaskName] = task_name
        self.inference_config_dict: Final[Mapping[str, Any]] = (
            inference_config_dict if inference_config_dict is not None else {}
        )
        self.session_wrapper_list: list[SessionWrapper] = []
        self.self_consistency_entry_list: list[SelfConsistencyEntry] = []
        self.current_self_consistency_entry: Optional[SelfConsistencyEntry] = None

    def _get_session_wrapper_list_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "session_wrapper_list.json")

    def _get_self_consistency_entry_list_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "self_consistency_entry_list.json")

    def _get_batch_size_manager_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "batch_size_manager.json")

    def restore_state(self) -> None:
        self.session_wrapper_list = [
            SessionWrapper.model_validate(session_info_dict)
            for session_info_dict in json.load(
                open(self._get_session_wrapper_list_state_path(), "r")
            )
        ]
        self.self_consistency_entry_list = [
            SelfConsistencyEntry.model_validate(entry_info_dict)
            for entry_info_dict in json.load(
                open(self._get_self_consistency_entry_list_state_path(), "r")
            )
        ]
        self.batch_size_manager.load_state(self._get_batch_size_manager_state_path())

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _language_model_dynamic_batch_inference(
        self,
        inference_phase: SelfConsistencyPhase,
        batch_chat_history: Sequence[ChatHistory],
    ) -> Sequence[ChatHistoryItem]:
        inference_result_list: list[ChatHistoryItem] = []
        current_batch_size = self.batch_size_manager.get(inference_phase)
        while True:
            for start_index in range(0, len(batch_chat_history), current_batch_size):
                batch_chat_history_slice = batch_chat_history[
                    start_index : start_index + current_batch_size
                ]
                try:
                    batch_inference_result = self.language_model.inference(
                        batch_chat_history_slice, self.inference_config_dict
                    )
                except Exception as e:
                    if (
                        isinstance(e, LanguageModelOutOfMemoryException)
                        and current_batch_size > 1
                    ):
                        current_batch_size //= 2
                        inference_result_list.clear()
                        break
                    caller_frame = inspect.stack()[1]
                    function_name = caller_frame.function
                    error_message = (
                        f"[GroupSelfConsistencyCallback.{function_name}()]: "
                        f"Error in self.language_model.inference() with batch size {current_batch_size}.\n"
                        f"{str(e)}"
                    )
                    SafeLogger.error(error_message)
                    return [
                        ChatHistoryItem(
                            role=Role.AGENT,
                            content=error_message,
                        )
                        for _ in range(len(batch_chat_history))
                    ]
                inference_result_list.extend(batch_inference_result)
            else:
                break
        self.batch_size_manager.record_usage(inference_phase, current_batch_size)
        self.batch_size_manager.update(inference_phase)
        return inference_result_list

    def _construct_relevance_judgement_prompt(self) -> str:
        prompt: str
        match self.task_name:
            case TaskName.DB_BENCH:
                requirement_str = """- Review both the past experience and the current question.
- Identify the SQL skills required for both the past experience and the current question. For example, If the question of the experience is : "Which customers have given a rating of 4 or higher? Return the customer ID, rating, and comments, limited to 5 entries.", Then, the skills involved in this question include: "limit_only", "where_single_condition".
- Determine if the current question shares similar SQL skills with the past experience."""
            case TaskName.OS_INTERACTION:
                requirement_str = """- Review both the past experience and the current question.
- Identity the commands required for both the past experience and the current question. For example, If the question of the experience is : "Copy the file '/home/user/report.log' to '/var/archive/', set group ownership to 'archivers', and ensure the group has read-only permissions.", Then, the skills involved in this question include: "chgrp", "chmod", "cp".
- Determine if the current question shares similar commands with the past experience.
"""
            case _:
                raise NotImplementedError()
        prompt = f"""Your task is to determine if a past experience is relevant to a new question based on the similarity of the required skills.

The given experience is as follows:
- Experience question:
{{experience_question}}
- Experience solution:
{{experience_solution}}
Now I will give you the current question.
- Current question:
{{current_question}}

Requirements:
{requirement_str}

Output format:
After the analysis, If you think the question and the experience are relevant, you need to give your judgement in the following format:
Answer: Relevant
If you think the question and the experience are irrelevant, you need to give your judgement in the following format:
Answer: Irrelevant
"""
        return prompt

    def _select_session_wrapper_by_sample_index(
        self, sample_index: SampleIndex
    ) -> SessionWrapper:
        for session_wrapper in self.session_wrapper_list:
            if session_wrapper.session.sample_index == sample_index:
                return session_wrapper
        raise RuntimeError(
            f"Cannot find the session wrapper with sample_index {sample_index}"
        )

    def _construct_relevance_info_dict(
        self, chat_history_info_list: Sequence[ChatHistoryInfo]
    ) -> Mapping[SampleIndex, RelevanceInfo]:
        relevance_info_dict: dict[SampleIndex, RelevanceInfo] = {}
        inference_batch_size = self.batch_size_manager.get(
            SelfConsistencyPhase.RELEVANCE_JUDGEMENT
        )
        current_relevant_sample_count = 0
        if self.group_count is None:
            target_relevant_sample_count = len(chat_history_info_list)
        else:
            target_relevant_sample_count = (
                self.group_count * self.sample_count_per_group
            )
        for start_chat_history_info_index in range(
            0,
            len(chat_history_info_list),
            inference_batch_size,
        ):
            batch_chat_history_info = chat_history_info_list[
                start_chat_history_info_index : start_chat_history_info_index
                + inference_batch_size
            ]
            batch_sample_index_list = [
                chat_history_info.sample_index_list[0]
                for chat_history_info in batch_chat_history_info
            ]
            batch_chat_history = [
                chat_history_info.chat_history
                for chat_history_info in batch_chat_history_info
            ]
            batch_inference_result = self._language_model_dynamic_batch_inference(
                SelfConsistencyPhase.RELEVANCE_JUDGEMENT, batch_chat_history
            )
            for sample_index, inference_result in zip(
                batch_sample_index_list, batch_inference_result
            ):
                relevant_flag = "Answer: Relevant" in inference_result.content
                irrelevant_flag = "Answer: Irrelevant" in inference_result.content
                if relevant_flag and not irrelevant_flag:
                    judgement = RelevanceJudgement.RELEVANT
                    current_relevant_sample_count += 1
                elif not relevant_flag and irrelevant_flag:
                    judgement = RelevanceJudgement.IRRELEVANT
                else:
                    judgement = RelevanceJudgement.UNKNOWN
                relevance_info_dict[sample_index] = RelevanceInfo(
                    raw_inference=inference_result.content,
                    judgement=judgement,
                )
            if current_relevant_sample_count >= target_relevant_sample_count:
                # Already have enough relevant samples
                break
        return relevance_info_dict

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        # region Preparation
        current_session_chat_history = callback_args.current_session.chat_history
        assert current_session_chat_history.get_value_length() == 3
        # user: requirement
        # agent: OK.
        # user: question
        question_chat_history_item = current_session_chat_history.get_item_deep_copy(-1)
        assert question_chat_history_item.role == Role.USER
        current_question = question_chat_history_item.content
        raw_prompt = self._construct_relevance_judgement_prompt()
        # endregion
        # region Construct chat_history_dict
        chat_history_info_list: list[ChatHistoryInfo] = []
        sorted_session_wrapper_list = sorted(self.session_wrapper_list, reverse=True)
        for session_wrapper in sorted_session_wrapper_list:
            processed_prompt = (
                raw_prompt.replace(
                    "{experience_question}", session_wrapper.experience_question
                )
                .replace("{experience_solution}", session_wrapper.experience_solution)
                .replace("{current_question}", current_question)
            )
            chat_history_info_list.append(
                ChatHistoryInfo(
                    chat_history=ChatHistory(
                        value=[
                            ChatHistoryItem(role=Role.USER, content=processed_prompt)
                        ]
                    ),
                    sample_index_list=[session_wrapper.session.sample_index],
                )
            )
        # endregion
        # region Judge the relevance of the experience by self.language_model
        # Disable the relevance judgement for now
        # relevance_info_dict = self._construct_relevance_info_dict(
        #     chat_history_info_list
        # )
        relevance_info_dict = {
            chat_history_info.sample_index_list[0]: RelevanceInfo(
                raw_inference="Always relevant.",
                judgement=RelevanceJudgement.RELEVANT,
            )
            for chat_history_info in chat_history_info_list
        }
        # endregion
        # region Set self.current_self_consistency_entry
        self.current_self_consistency_entry = SelfConsistencyEntry(
            current_session_index=callback_args.current_session.sample_index,
            relevance_info_dict=relevance_info_dict,
            inference_outcome_list=[],
            current_session_to_priority_dict={
                session_wrapper.session.sample_index: session_wrapper.priority
                for session_wrapper in self.session_wrapper_list
            },
        )
        # endregion

    def _construct_sorted_utilized_sample_index_list(self) -> Sequence[SampleIndex]:
        assert self.current_self_consistency_entry is not None
        utilized_sample_wrapper_list: list[SessionWrapper] = []
        for (
            sample_index,
            relevance_info,
        ) in self.current_self_consistency_entry.relevance_info_dict.items():
            if not relevance_info.judgement == RelevanceJudgement.RELEVANT:
                continue
            session_wrapper = self._select_session_wrapper_by_sample_index(sample_index)
            utilized_sample_wrapper_list.append(session_wrapper)
        # Although utilized_sample_wrapper_list is already sorted, we sort it again to ensure the order
        utilized_sample_wrapper_list.sort(reverse=True)
        sorted_utilized_sample_index_list = [
            session_wrapper.session.sample_index
            for session_wrapper in utilized_sample_wrapper_list
        ]
        if self.group_count is not None:
            sorted_utilized_sample_index_list = sorted_utilized_sample_index_list[
                : self.group_count * self.sample_count_per_group
            ]
        return sorted_utilized_sample_index_list

    def _construct_experience_utilization_inference_prompt(self) -> str:
        prompt = f"""{TASK_REQUIREMENT_DICT[self.task_name]}

{{experience_utilization_prompt}}

Now, I will give you the question that you need to solve."""
        return prompt

    def _extract_action(self, agent_raw_inference: str) -> Optional[str]:
        match self.task_name:
            case TaskName.DB_BENCH:
                try:
                    db_bencb_parser_result = DBBench._parse_agent_response(  # noqa
                        agent_raw_inference
                    )
                except:  # noqa
                    return None
                match db_bencb_parser_result.action:
                    case AgentAction.EXECUTE:
                        sql = db_bencb_parser_result.content or ""
                        try:
                            ast = sqlglot.parse_one(sql)
                            reconstructed_sql = ast.sql()
                        except:  # noqa
                            reconstructed_sql = sql
                        return f"Action: Operation\n```sql\n{reconstructed_sql}\n```"
                    case AgentAction.FINISH:
                        return f"Action: Answer\nFinal Answer: {db_bencb_parser_result.content}"
                    case _:
                        return None
            case TaskName.OS_INTERACTION:
                try:
                    os_interaction_parser_result = OSInteraction._parse_agent_response(
                        agent_raw_inference
                    )  # noqa
                except:  # noqa
                    return None
                match os_interaction_parser_result.action:
                    case AgentAction.EXECUTE:
                        command = os_interaction_parser_result.content or ""
                        return f"Act: bash\n```bash\n{command}\n```"
                    case AgentAction.FINISH:
                        return f"Act: finish"
                    case _:
                        return None
            case _:
                raise NotImplementedError()

    def _select_action_from_candidate_action_list(
        self, candidate_action_list: Sequence[str], session_chat_history: ChatHistory
    ) -> tuple[Optional[str], str]:  # selected action, selected_reason
        _ = session_chat_history.pop(-1)
        prompt: str = """The response of your last action is: {original_user_content}. Now, Please determine the best action for me to take from the following options:
{action_selection_str}
Analyze the choices carefully, explain your reasoning, and then select the best option.
Format your selection as follows (e.g., if choosing option A):
Selection: A
"""
        action_selection_str: str = ""
        for action_index, candidate_action in enumerate(candidate_action_list):
            action_selection_str += f"{chr(65 + action_index)}. {candidate_action}\n"
        original_user_content = session_chat_history.get_item_deep_copy(-1).content
        prompt.replace("{action_selection_str}", action_selection_str)
        prompt.replace("{original_user_content}", original_user_content)
        session_chat_history.set(-1, ChatHistoryItem(role=Role.USER, content=prompt))
        try:
            self.language_model.inference(
                [session_chat_history], self.inference_config_dict
            )
        except Exception as e:
            SafeLogger.error(
                f"[GroupSelfConsistencyCallback._select_action_from_candidate_action_list()]: "
                f"Error in inferencing using the language model.\n"
                f"{str(e)}"
            )
            return None, "Error in agent inference"
        match = re.search(
            r"Selection: ([A-Z])", session_chat_history.get_item_deep_copy(-1).content
        )
        if match is None:
            return None, "Cannot extract the selection"
        selected_action_index = ord(match.group(1)) - 65
        if (
            not isinstance(selected_action_index, int)
            or selected_action_index < 0
            or selected_action_index >= len(candidate_action_list)
        ):
            return None, "Invalid selection"
        return (
            candidate_action_list[selected_action_index],
            "Selected by the self.language_model",
        )

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        assert self.current_self_consistency_entry is not None
        sorted_utilized_sample_index_list = (
            self._construct_sorted_utilized_sample_index_list()
        )
        raw_prompt = self._construct_experience_utilization_inference_prompt()
        # region Construct chat_history_info_list
        chat_history_info_list: list[ChatHistoryInfo] = []
        for start_sample_index in range(
            0, len(sorted_utilized_sample_index_list), self.sample_count_per_group
        ):
            # region Construct processed_prompt
            processed_prompt: str
            if self.sample_count_per_group == 1:
                session_wrapper = self._select_session_wrapper_by_sample_index(
                    sorted_utilized_sample_index_list[start_sample_index]
                )
                experience_utilization_prompt = """Before giving you the question that you need to solve, I will provide you with an experience. You can use the experience to help you solve the question.
- Experience question:
{experience_question}
- Experience solution:
{experience_solution}"""
                experience_utilization_prompt = experience_utilization_prompt.replace(
                    "{experience_question}", session_wrapper.experience_question
                ).replace("{experience_solution}", session_wrapper.experience_solution)
                processed_prompt = raw_prompt.replace(
                    "{experience_utilization_prompt}", experience_utilization_prompt
                )
            else:
                sorted_session_wrapper_list = [
                    self._select_session_wrapper_by_sample_index(sample_index)
                    for sample_index in sorted_utilized_sample_index_list[
                        start_sample_index : start_sample_index
                        + self.sample_count_per_group
                    ]
                ]
                experience_utilization_prompt = """Before giving you the question that you need to solve, I will provide you with same experience. You can use these experience to help you solve the question.
{concatenated_experience}"""
                concatenated_experience = ""
                for experience_index, session_wrapper in enumerate(
                    sorted_session_wrapper_list
                ):
                    concatenated_experience += f"""
Experience {experience_index + 1}:
- Experience question:
{session_wrapper.experience_question}
- Experience solution:
{session_wrapper.experience_solution}
"""
                concatenated_experience = (
                    concatenated_experience.strip()
                )  # Remove the leading and trailing whitespaces
                experience_utilization_prompt = experience_utilization_prompt.replace(
                    "{concatenated_experience}", concatenated_experience
                )
                processed_prompt = raw_prompt.replace(
                    "{experience_utilization_prompt}", experience_utilization_prompt
                )
            # endregion
            # region Construct chat_history
            chat_history_deep_copy = (
                callback_args.current_session.chat_history.model_copy(deep=True)
            )
            _ = chat_history_deep_copy.pop(-1)  # Remove the newest agent response
            chat_history_deep_copy.set(
                0, ChatHistoryItem(role=Role.USER, content=processed_prompt)
            )  # Replace the first user prompt with the processed_prompt
            # endregion
            chat_history_info_list.append(
                ChatHistoryInfo(
                    chat_history=chat_history_deep_copy,
                    sample_index_list=sorted_utilized_sample_index_list[
                        start_sample_index : start_sample_index
                        + self.sample_count_per_group
                    ],
                )
            )
        # endregion
        # region Inference by self.language_model
        inference_result_list: list[ChatHistoryItem] = []
        inference_batch_size = self.batch_size_manager.get(
            SelfConsistencyPhase.EXPERIENCE_UTILIZATION
        )
        for start_chat_history_info_index in range(
            0,
            len(chat_history_info_list),
            inference_batch_size,
        ):
            batch_chat_history_info = chat_history_info_list[
                start_chat_history_info_index : start_chat_history_info_index
                + inference_batch_size
            ]
            batch_chat_history = [
                chat_history_info.chat_history
                for chat_history_info in batch_chat_history_info
            ]
            batch_inference_result = self._language_model_dynamic_batch_inference(
                SelfConsistencyPhase.EXPERIENCE_UTILIZATION, batch_chat_history
            )
            inference_result_list.extend(batch_inference_result)
        # endregion
        # region Construct group_info_list
        group_info_list: list[GroupInfo] = []
        # region Add the original agent response
        original_inference_content = (
            callback_args.current_session.chat_history.get_item_deep_copy(-1).content
        )
        group_info_list.append(
            GroupInfo(
                experience_sample_index_list=None,  # No experience is provided
                raw_inference=original_inference_content,
                extracted_action=self._extract_action(original_inference_content),
                group_index=len(group_info_list),
            )
        )
        # endregion
        # region Add responses from inference_result_list
        for chat_history_info, inference_result in zip(
            chat_history_info_list, inference_result_list
        ):
            group_info_list.append(
                GroupInfo(
                    experience_sample_index_list=chat_history_info.sample_index_list,
                    raw_inference=inference_result.content,
                    extracted_action=self._extract_action(inference_result.content),
                    group_index=len(group_info_list),
                )
            )
        # endregion  # noqa
        # endregion
        # region Construct action_to_group_index_list_dict and candidate_action_list
        action_to_group_index_list_dict: dict[str, list[int]] = {}
        maximum_action_count: int = 1
        for group_info in group_info_list:
            action = group_info.extracted_action
            if action is None:
                continue
            if action not in action_to_group_index_list_dict:
                action_to_group_index_list_dict[action] = []
            action_to_group_index_list_dict[action].append(group_info.group_index)
            maximum_action_count = max(
                maximum_action_count, len(action_to_group_index_list_dict[action])
            )
        candidate_action_list = [
            action
            for action, group_index_list in action_to_group_index_list_dict.items()
            if len(group_index_list) == maximum_action_count
        ]
        # endregion
        # region Select action, maintain current_session.chat_history, self.current_self_consistency_entry
        if len(candidate_action_list) == 0:
            # current_session is not modified
            self.current_self_consistency_entry.inference_outcome_list.append(
                SelfConsistencyInferenceOutcome(
                    group_info_list=group_info_list,
                    action_to_sample_index_list_dict=action_to_group_index_list_dict,
                    selected_action=None,
                    selected_reason="Cannot extract any action",
                )
            )
        elif len(candidate_action_list) == 1:
            selected_action_str = candidate_action_list[0]
            selected_group_info = group_info_list[
                action_to_group_index_list_dict[selected_action_str][0]
            ]
            callback_args.current_session.chat_history.set(
                -1,
                ChatHistoryItem(
                    role=Role.AGENT, content=selected_group_info.raw_inference
                ),
            )
            self.current_self_consistency_entry.inference_outcome_list.append(
                SelfConsistencyInferenceOutcome(
                    group_info_list=group_info_list,
                    action_to_sample_index_list_dict=action_to_group_index_list_dict,
                    selected_action=selected_action_str,
                    selected_reason="Only one action is extracted",
                )
            )
        else:
            # Ugly implementation, refactor in the future.
            original_agent_action = group_info_list[0].extracted_action
            if original_agent_action in candidate_action_list:
                assert isinstance(original_agent_action, str)  # Type narrowing
                candidate_action_list.remove(original_agent_action)
            if len(candidate_action_list) == 1:
                selected_action_str = candidate_action_list[0]
                selected_group_info = group_info_list[
                    action_to_group_index_list_dict[selected_action_str][0]
                ]
                callback_args.current_session.chat_history.set(
                    -1,
                    ChatHistoryItem(
                        role=Role.AGENT, content=selected_group_info.raw_inference
                    ),
                )
                self.current_self_consistency_entry.inference_outcome_list.append(
                    SelfConsistencyInferenceOutcome(
                        group_info_list=group_info_list,
                        action_to_sample_index_list_dict=action_to_group_index_list_dict,
                        selected_action=selected_action_str,
                        selected_reason="After removing the original agent action, only one action is extracted",
                    )
                )
                return
            chat_history_deep_copy = (
                callback_args.current_session.chat_history.model_copy(deep=True)
            )
            selected_action, selected_reason = (
                self._select_action_from_candidate_action_list(
                    candidate_action_list, chat_history_deep_copy
                )
            )
            if selected_action is not None:
                selected_group_info = group_info_list[
                    action_to_group_index_list_dict[selected_action][0]
                ]
                callback_args.current_session.chat_history.set(
                    -1,
                    ChatHistoryItem(
                        role=Role.AGENT, content=selected_group_info.raw_inference
                    ),
                )
            self.current_self_consistency_entry.inference_outcome_list.append(
                SelfConsistencyInferenceOutcome(
                    group_info_list=group_info_list,
                    action_to_sample_index_list_dict=action_to_group_index_list_dict,
                    selected_action=selected_action,
                    selected_reason=selected_reason,
                )
            )
        # endregion

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        # region Maintain self.session_wrapper_list
        assert self.current_self_consistency_entry is not None
        self.self_consistency_entry_list.append(self.current_self_consistency_entry)
        # endregion
        # region Maintain session_wrapper priority in self.session_wrapper_list
        # The implementation of this part can be improved in the future
        sorted_utilized_sample_index_list = (
            self._construct_sorted_utilized_sample_index_list()
        )
        if (
            callback_args.current_session.evaluation_record.outcome
            != SessionEvaluationOutcome.CORRECT
        ):
            for session_wrapper in self.session_wrapper_list:
                if (
                    session_wrapper.session.sample_index
                    in sorted_utilized_sample_index_list
                ):
                    session_wrapper.priority -= 0  # Disable the priority update for now
            return
        else:
            for session_wrapper in self.session_wrapper_list:
                if (
                    session_wrapper.session.sample_index
                    in sorted_utilized_sample_index_list
                ):
                    session_wrapper.priority += 0  # Disable the priority update for now
        # endregion
        # region Maintain self.session_wrapper_list
        chat_history = callback_args.current_session.chat_history
        experience_question = chat_history.get_item_deep_copy(2).content
        agent_role_dict = self.language_model.role_dict
        # Skip the first 3 items, which are
        # - user: requirement
        # - agent: OK.
        # - user: question
        experience_solution = chat_history.get_value_str(
            agent_role_dict, start_index=3, end_index=None
        )
        priority = 0
        for session_wrapper in self.session_wrapper_list:
            if (
                session_wrapper.session.sample_index
                in sorted_utilized_sample_index_list
            ):
                priority = max(priority, session_wrapper.priority)
        self.session_wrapper_list.append(
            SessionWrapper(
                session=callback_args.current_session.model_copy(deep=True),
                experience_question=experience_question,
                experience_solution=experience_solution,
                priority=priority,
                created_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        # endregion
        # region Clean up
        # Some functions that are called below may also need to access self.current_self_consistency_entry
        self.current_self_consistency_entry = None
        # endregion

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        json.dump(
            [session.model_dump() for session in self.session_wrapper_list],
            open(self._get_session_wrapper_list_state_path(), "w"),  # noqa
            indent=2,
        )
        json.dump(
            [entry.model_dump() for entry in self.self_consistency_entry_list],
            open(self._get_self_consistency_entry_list_state_path(), "w"),  # noqa
            indent=2,
        )
        self.batch_size_manager.dump_state(self._get_batch_size_manager_state_path())
