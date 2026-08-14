from typing import final, Optional, Any, TypeVar, Generic, Sequence
from abc import ABC, abstractmethod

from pydantic import BaseModel
from enum import StrEnum
import inspect
import warnings

from src.typings import (
    SampleIndex,
    SampleStatus,
    TaskUnknownException,
    TaskReleaseException,
    ContinualAgentBenchException,
    TaskEnvironmentException,
    Session,
    TaskName,
    Role,
    SessionEvaluationOutcome,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.utils import SafeLogger
from src.factories.chat_history_item import ChatHistoryItemFactory


class SkillUtility:
    _SKILL_TO_LEVEL_DICT: dict[str, int] = {}

    @classmethod
    @final
    def is_valid_skill(cls, skill: str) -> bool:
        return skill in cls._SKILL_TO_LEVEL_DICT.keys()

    @classmethod
    @final
    def get_skill_level(cls, skill: str) -> int:
        return cls._SKILL_TO_LEVEL_DICT[skill]

    @classmethod
    @final
    def get_all_skill_list(cls) -> list[str]:
        return list(cls._SKILL_TO_LEVEL_DICT.keys())

    @classmethod
    @final
    def get_skill_level_list(cls) -> list[int]:
        skill_level_list = list(set(cls._SKILL_TO_LEVEL_DICT.values()))
        assert min(skill_level_list) == 0  # Do not remove this line
        return skill_level_list


class DatasetItem(BaseModel):
    @abstractmethod
    def get_skill_list(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_difficulty_level(self) -> int:
        raise NotImplementedError()

    def get_effective_skill_list(self) -> list[str]:
        # By default, all skills are deemed as effective.
        # It can also be overridden by the subclass, such as the most difficult skill.
        return self.get_skill_list()


class TaskInterface(ABC):
    @abstractmethod
    def get_sample_index_list(self) -> list[SampleIndex]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, session: Session) -> None:
        raise NotImplementedError()

    @abstractmethod
    def interact(self, session: Session) -> None:
        raise NotImplementedError()

    @abstractmethod
    def complete(self, session: Session) -> None:
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def calculate_metric(
        self,
        session_partial_list: Sequence[SessionMetricCalculationPartial],
    ) -> MetricDict:
        raise NotImplementedError()


class AgentAction(StrEnum):
    EXECUTE = "execute"
    FINISH = "finish"
    INVALID = "invalid"


class AgentResponseParserResult(BaseModel):
    action: AgentAction
    content: Optional[str]
    finish_reason: Optional[str]


# https://stackoverflow.com/a/78771247
T = TypeVar("T")
DatasetItemSubclass = TypeVar("DatasetItemSubclass", bound=DatasetItem)


class Task(TaskInterface, Generic[DatasetItemSubclass]):
    def __init__(
        self,
        task_name: TaskName,
        chat_history_item_factory: ChatHistoryItemFactory,
        max_round: int,
    ):
        """
        I do not think it is a good idea to make skill_utility_cls a data member currently, since it will greatly
        reduce the flexibility of the task.
        """
        super().__init__()
        self.task_name = task_name
        self.chat_history_item_factory = chat_history_item_factory
        self.current_sample_index: Optional[SampleIndex] = None
        self.max_round = max_round
        self.current_round = 0
        self.__dataset: Optional[dict[SampleIndex, DatasetItemSubclass]] = None
        self.__current_dataset_item: Optional[DatasetItemSubclass] = None

    @final
    def _set_dataset(self, dataset: dict[SampleIndex, DatasetItemSubclass]) -> None:
        # Must be called in the __init__ method of the subclass
        assert self.__dataset is None
        self.__dataset = dataset

    @final
    def _get_current_dataset_item(self) -> DatasetItemSubclass:
        assert self.__current_dataset_item is not None
        return self.__current_dataset_item

    @final
    def __get_dataset_item(self, sample_index: SampleIndex) -> DatasetItemSubclass:
        caller_frame = inspect.stack()[1]
        expected_function_name_list = [
            self._calculate_metric_based_on_skill.__name__,
            self._calculate_metric_based_on_difficulty_level.__name__,
        ]
        if caller_frame.function not in expected_function_name_list:
            allowed_function_str = ""
            for expected_function_name in expected_function_name_list:
                allowed_function_str += f"Task.{expected_function_name}, "
            allowed_function_str = allowed_function_str[:-2] + "."
            warnings.warn(
                f"The function is only expected to be called by {allowed_function_str}",
                RuntimeWarning,
            )
        # The function will not check the existence of the sample index in the dataset,
        # since it is a very low-level function.
        assert self.__dataset is not None
        return self.__dataset[sample_index]

    @abstractmethod
    def _get_default_task_output(self) -> dict[str, Optional[str]]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _parse_agent_response(agent_response: str) -> AgentResponseParserResult:
        """
        The method must be implemented in the subclass.
        It is used to parse the agent response by series of regular expressions
        and return the parsed result.
        The parsed result may be a very complicated object, so it is marked as Any.
        Currently, I do not think there exists a way to enforce the return type.
        """
        raise NotImplementedError()

    @staticmethod
    @final
    def _calculate_correct_rate(
        count_dict: dict[T, int], correct_count_dict: dict[T, int]
    ) -> dict[T, float]:
        correct_rate_dict: dict[T, int | float] = {}
        for key in count_dict.keys():
            if count_dict[key] == 0 or key not in correct_count_dict:
                correct_rate_dict[key] = -1
            else:
                correct_rate_dict[key] = correct_count_dict[key] / count_dict[key]
        return correct_rate_dict

    @final
    def _calculate_metric_based_on_skill(
        self,
        skill_utility_cls: type[SkillUtility],
        session_partial_list: Sequence[SessionMetricCalculationPartial],
    ) -> dict[str, dict[str, float]]:
        count_dict = {key: 0 for key in skill_utility_cls.get_all_skill_list()}
        correct_count_dict = {key: 0 for key in skill_utility_cls.get_all_skill_list()}
        effective_count_dict = {
            key: 0 for key in skill_utility_cls.get_all_skill_list()
        }
        effective_correct_count_dict = {
            key: 0 for key in skill_utility_cls.get_all_skill_list()
        }
        for session_partial in session_partial_list:
            sample_index = session_partial.sample_index
            dataset_item: DatasetItemSubclass = self.__get_dataset_item(sample_index)
            for skill in dataset_item.get_skill_list():
                # region Handle total count
                count_dict[skill] += 1
                if skill in dataset_item.get_effective_skill_list():
                    effective_count_dict[skill] += 1
                # endregion
                # region Handle correct count
                if (
                    session_partial.evaluation_record.outcome
                    == SessionEvaluationOutcome.CORRECT
                ):
                    correct_count_dict[skill] += 1
                    if skill in dataset_item.get_effective_skill_list():
                        effective_correct_count_dict[skill] += 1
                # endregion
        skill_correct_rate_dict = Task._calculate_correct_rate(
            count_dict, correct_count_dict
        )
        effective_skill_correct_rate_dict = Task._calculate_correct_rate(
            effective_count_dict, effective_correct_count_dict
        )
        skill_metric_dict: dict[str, dict[str, float]] = {
            "count_dict": dict(count_dict),
            "correct_count_dict": dict(correct_count_dict),
            "correct_rate_dict": skill_correct_rate_dict,
            "effective_count_dict": dict(effective_count_dict),
            "effective_correct_count_dict": dict(effective_correct_count_dict),
            "effective_correct_rate_dict": effective_skill_correct_rate_dict,
        }
        return skill_metric_dict

    @final
    def _calculate_metric_based_on_difficulty_level(
        self,
        session_partial_list: Sequence[SessionMetricCalculationPartial],
    ) -> dict[str, dict[str, float]]:
        # region Preparation
        difficulty_level_set: set[int] = set()
        for session_partial in session_partial_list:
            # Use for-loop instead of list comprehension to mute warnings.
            difficulty_level: int = self.__get_dataset_item(
                session_partial.sample_index
            ).get_difficulty_level()
            difficulty_level_set.add(difficulty_level)
        difficulty_level_list: list[int] = sorted(list(difficulty_level_set))
        count_dict = {str(key): 0 for key in difficulty_level_list}
        # endregion
        correct_count_dict = {str(key): 0 for key in difficulty_level_list}
        for session_partial in session_partial_list:
            sample_index = session_partial.sample_index
            dataset_item: DatasetItemSubclass = self.__get_dataset_item(sample_index)
            difficulty_level = dataset_item.get_difficulty_level()
            difficulty_level_str = str(difficulty_level)
            count_dict[difficulty_level_str] += 1
            if (
                session_partial.evaluation_record.outcome
                == SessionEvaluationOutcome.CORRECT
            ):
                correct_count_dict[difficulty_level_str] += 1
        sample_level_correct_rate_dict = Task._calculate_correct_rate(
            count_dict, correct_count_dict
        )
        difficulty_level_metric_dict: dict[str, dict[str, float]] = {
            "count_dict": dict(count_dict),
            "correct_count_dict": dict(correct_count_dict),
            "correct_rate_dict": sample_level_correct_rate_dict,
        }
        return difficulty_level_metric_dict

    @staticmethod
    def _calculate_overall_metric(
        session_partial_list: Sequence[SessionMetricCalculationPartial],
    ) -> dict[str, dict[str, float]]:
        """
        The method can be overridden in the subclass, if necessary.
        """
        # region Record the number of sessions
        session_count = len(session_partial_list)
        overall_metric_dict: dict[str, dict[str, float]] = {
            "basic": {"session_count": float(session_count)},
        }
        if session_count == 0:
            return overall_metric_dict
        # endregion
        # region Calculate the rate of each SessionEvaluationOutcome
        evaluation_outcome_metric_dict = {}
        for evaluation_outcome in SessionEvaluationOutcome:
            outcome_count = len(
                [
                    session
                    for session in session_partial_list
                    if session.evaluation_record.outcome == evaluation_outcome
                ]
            )
            evaluation_outcome_metric_dict[str(evaluation_outcome)] = (
                outcome_count / session_count
            )
        overall_metric_dict["evaluation_outcome"] = evaluation_outcome_metric_dict
        # endregion
        # region Calculate the rate of each SampleStatus
        sample_status_metric_dict = {}
        for sample_status in SampleStatus:
            status_count = len(
                [
                    session
                    for session in session_partial_list
                    if session.sample_status == sample_status
                ]
            )
            sample_status_metric_dict[str(sample_status)] = status_count / session_count
        overall_metric_dict["sample_status"] = sample_status_metric_dict
        # endregion
        return overall_metric_dict

    @final
    def get_sample_index_list(self) -> list[SampleIndex]:
        assert self.__dataset is not None
        return list(self.__dataset.keys())

    @final
    def reset(self, session: Session) -> None:
        """
        Reset the task state using the provided session.
        """
        # Do Validation and manage the state of the task
        assert session.sample_status == SampleStatus.INITIAL
        assert session.task_name == self.task_name
        assert self.current_sample_index is None
        assert self.__current_dataset_item is None
        self.current_sample_index = session.sample_index
        self.current_round = 0
        assert self.__dataset is not None
        self.__current_dataset_item = self.__dataset[session.sample_index]
        session.sample_status = SampleStatus.RUNNING
        try:
            self._reset(session)
        except ContinualAgentBenchException as e:
            session.finish_reason = str(e)
            if isinstance(e, TaskEnvironmentException):
                session.sample_status = SampleStatus.TASK_ENVIRONMENT_ERROR
            else:
                raise TypeError(
                    f"Please handle {e.__class__.__name__} in Task.reset()."
                )
        except Exception as e:
            _ = TaskUnknownException(str(e))  # Record the exception
            session.sample_status = SampleStatus.TASK_UNKNOWN_ERROR
            session.finish_reason = str(TaskUnknownException.from_exception(e))

    @abstractmethod
    def _reset(self, session: Session) -> None:
        """
        Internal method to reset the task. Should be overridden by subclasses.
        """
        raise NotImplementedError()

    @final
    def interact(self, session: Session) -> None:
        """
        Handle interaction for the task using the given session.
        """
        # region Do Validation and manage the state of the task
        assert (
            session.sample_status == SampleStatus.RUNNING
            and session.finish_reason is None
        ) or session.sample_status.is_agent_inference_process_abnormal()
        assert session.sample_index == self.current_sample_index
        assert session.task_name == self.task_name
        assert session.chat_history.get_item_deep_copy(-1).role == Role.AGENT
        assert session.task_output is None
        # endregion
        try:
            # region Handle SampleStatus.AGENT_CONTEXT_LIMIT and SampleStatus.AGENT_UNKNOWN_ERROR
            if session.sample_status.is_agent_inference_process_abnormal():
                session.task_output = self._get_default_task_output()
                return
            # endregion
            # region Check whether the session reaches the round limit
            if not self.current_round < self.max_round:
                session.sample_status = SampleStatus.TASK_LIMIT_REACHED
                session.task_output = self._get_default_task_output()
                session.finish_reason = (
                    f"Task limit reached. The limit is {self.max_round}."
                )
                return
            self.current_round += 1
            # endregion
            # region Finally, the agent is allowed to interact with the task
            self._interact(session)
            # endregion
        except ContinualAgentBenchException as e:
            session.finish_reason = str(e)
            if isinstance(e, TaskEnvironmentException):
                session.sample_status = SampleStatus.TASK_ENVIRONMENT_ERROR
            elif isinstance(e, TaskUnknownException):
                session.sample_status = SampleStatus.TASK_UNKNOWN_ERROR
            else:
                raise TypeError(
                    f"Please handle {e.__class__.__name__} in Task.inference()."
                )
        except Exception as e:
            _ = TaskUnknownException(str(e))  # Record the exception
            session.sample_status = SampleStatus.TASK_UNKNOWN_ERROR
            session.finish_reason = str(TaskUnknownException.from_exception(e))

    @abstractmethod
    def _interact(self, session: Session) -> None:
        """
        Handle interaction.
        Pseudocode:
        ```
        parsed_result = self._parse_agent_response(last_agent_response)
        match parsed_result.response_type:  # It should be a StrEnum
            case ResponseType.Type1:
                pass  # Interact with the environment
            case ResponseType.Type2:
                pass  # Interact with the environment
        ```
        """
        raise NotImplementedError()

    @final
    def complete(self, session: Session) -> None:
        """
        Complete the task and perform necessary finalization using the session.
        """
        # Do validation
        assert session.sample_status != SampleStatus.RUNNING
        assert session.sample_index == self.current_sample_index
        assert session.task_name == self.task_name
        assert self.current_sample_index is not None
        # Following statement is NOT allowed, since the method of getting default task output may also throw exceptions.
        #   assert session.task_output is not None
        try:
            self._complete(session)
        except ContinualAgentBenchException as e:
            if session.finish_reason is None:
                session.finish_reason = str(e)
            else:
                session.finish_reason += (
                    f" Another error happens in Task._complete(): {str(e)}"
                )
            session.evaluation_record.outcome = SessionEvaluationOutcome.UNKNOWN
        except Exception as e:
            _ = TaskUnknownException(str(e))  # Record the exception
            session.evaluation_record.outcome = SessionEvaluationOutcome.UNKNOWN
            _ = TaskUnknownException.from_exception(
                e
            )  # When debugging, this line can throw exception
        self.current_sample_index = None
        self.current_round = 0
        self.__current_dataset_item = None
        assert session.evaluation_record.outcome != SessionEvaluationOutcome.UNSET

    @abstractmethod
    def _complete(self, session: Session) -> None:
        """
        Internal method to complete the task. Should be overridden by subclasses.
        """
        raise NotImplementedError()

    @final
    def release(self) -> None:
        """
        Final release method that handles the cleanup process.
        """
        try:
            # Attempt to perform the release operation by calling the internal method
            self._release()
        except ContinualAgentBenchException as e:
            if isinstance(e, TaskReleaseException):
                # If the exception is a TaskReleaseException, log a warning.
                # If you want to see the trace of exception, raise it here.
                SafeLogger.error(f"Task release failed due to following exception: {e}")
            else:
                raise TypeError(
                    f"Please handle {e.__class__.__name__} in Task.release()."
                )
        except Exception as e:
            SafeLogger.error(f"Task release failed due to following exception: {e}")

    @abstractmethod
    def _release(self) -> None:
        """
        Internal release method intended to be overridden by subclasses.
        If the task do not need to release any resources, implement it by "pass".
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        """
        Calculate and return metrics based on a list of sessions.
        Past a list with one element to calculate the metric for a single sample.
        The most simple implementation is:
        ```
        return self._calculate_overall_metric(session_list)
        ```
        """
        raise NotImplementedError()
